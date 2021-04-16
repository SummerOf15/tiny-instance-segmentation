#!/bin/sh

source /volume/USERSTORE/lee_jn/miniconda3/bin/activate p_detector

export CUDA_HOME=/

DIR=/volume/USERSTORE/lee_jn/p_detector

TORCH_HOME=$DIR ; python $DIR/tools/sam_detection.py --num-gpus 8 --is-detr True --dist-url tcp://127.0.0.1:52111 --config-file $DIR/configs/sam/detr_resnet50_256_6_6_torchvision.yaml OUTPUT_DIR /home_local/lee_jn/experiments/samvr/detr_resnet50aug_finetune SOLVER.IMS_PER_BATCH 32 DATA_DIR /home_local/lee_jn/data_sam trainaugment True MODEL.WEIGHTS "/home_local/lee_jn/experiments/samvr/detr_resnet50aug_finetune/converted_model.pth"
