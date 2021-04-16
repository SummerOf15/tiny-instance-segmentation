'''
Training and evaluation scripts for sam-vr object detection module. Simple evaluations are also included.

To train: python sam_rcnn.py --num_gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 --config-file PATH_YAML_FILE

E.g. see logs/ scripts!

To evaluate: python sam_rcnn.py --config-file PATH_YAML_FILE --eval-only MODEL_CHECKPOINT

E.g python sam_rcnn.py --config-file ../configs/sam/faster_rcnn_resnet50_c4.yaml --eval-only MODEL.WEIGHTS /home_local/lee_jn/experiments/sam/sam_rcnn_resnet50_exp1/model_final.pth DATA_DIR /home_local_nvme/lee_jn/bagfiles/dataset

'''
import logging
import os
import torch
import detectron2.utils.comm as comm
import detectron2.data.transforms as T

from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from p_detector.dataset import register_samvr_voc
from p_detector.configs import add_samvr_config

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    '''
    Test script from detectron 2
    '''
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        
        def get_evaluator(cfg, dataset_name, output_folder=None):
            """
            Using the pascal voc data-set format only.
            
            Return pascal voc detection evaluator.
            """
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, tasks=("bbox", ), distributed=False, output_dir=output_folder)
        
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    '''
    Training script from detectron 2
    '''
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    if cfg.trainaugment:
        train_augmentations = [
            T.RandomBrightness(0.5, 2),
            T.RandomContrast(0.5, 2),
            T.RandomSaturation(0.5, 2),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),]
        data_loader = build_detection_train_loader(cfg, 
                      mapper=DatasetMapper(cfg, 
                                           is_train=True, 
                                           augmentations=train_augmentations))
    else:
        data_loader = build_detection_train_loader(cfg)
    
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    
    1. Data-set into the configuration.
    2. Configuration files in the configs folder.
    
    Argument configs:
    cfg -> used to build the model through build_model(cfg)
    cfg.OUTPUT_DIR -> checkpoint directory
    cfg.MODEL.WEIGHTS
    cfg.DATASETS.TRAIN = ("fruits_nuts",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS
    cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.BASE_LR
    cfg.SOLVER.MAX_ITER
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES
    """
    cfg = get_cfg()
    add_samvr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    # detectron2 config files
    cfg = setup(args)
    
    # register sam vr data-set using voc pascal format
    register_samvr_voc(name="samvr_2007_train", dirname=cfg.DATA_DIR, split='train', year='2007')
    register_samvr_voc(name="samvr_2007_test", dirname=cfg.DATA_DIR, split='test', year='2007')
    #MetadataCatalog.get("samvr_2007_train").year = 2007
    #MetadataCatalog.get("samvr_2007_test").year = 2007
    
    # build and train/eval the model
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
