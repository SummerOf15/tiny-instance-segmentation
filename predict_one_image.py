'''
predict the bounding box for one image and visualize the results
'''
from detectron2.config import get_cfg
import cv2
from detectron2.engine import DefaultPredictor
from p_detector.configs import add_tuft_config, add_detr_config
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def setup(detr=False):
    """
    Create configs and perform basic setups.
    
    1. Data-set into the configuration.
    2. Configuration files in the configs folder.
    """
    cfg = get_cfg()
    if detr:
        add_detr_config(cfg)
    else:
        add_tuft_config(cfg)
    cfg.merge_from_file("/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/configs/haicu/faster_rcnn_resnet50_fpn.yaml")
    cfg.MODEL.WEIGHTS = "/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/experiments/haicu/res50fpn/model_0039999.pth"
    cfg.freeze()
    return cfg


def detect_tufts(image_path, draw=False):
    img=cv2.imread(image_path)
    cfg = setup()

    model=DefaultPredictor(cfg)
    outputs = model(img)
    print(outputs)
    if draw:
        v= Visualizer(img[:, :, ::-1],
                        metadata=MetadataCatalog.get("tufts"), 
                        scale=0.5
            )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("result",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    return outputs


if __name__=="__main__":
    test_image_path="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/JPEGImages/DSC_2420.JPG"
    detection_results=detect_tufts(test_image_path)
    
