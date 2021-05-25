'''
predict the bounding box for one image and visualize the results
'''
from detectron2.config import get_cfg
import cv2
from detectron2.engine import DefaultPredictor
from p_detector.configs import add_tuft_config, add_detr_config
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from classification.affine_matching.affine_matching import predict_classification, parse_xml
import numpy as np
import os


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


def detect_tufts_one_image(image_path, draw=False):
    img=cv2.imread(image_path)
    cfg = setup()

    model=DefaultPredictor(cfg)
    outputs = model(img)
    if draw:
        detection_results=outputs["instances"].to("cpu")
        visualize_results(img,detection_results)
    return outputs



def calc_center_from_box(box_array):
    """calculate center point of boxes

    Args:
        box_array (array): N*4 [left_top_x, left_top_y, right_bottom_x, right_bottom_y]
    
    Returns:
        array N*2: center points array [x, y]
    """
    center_array=[]
    for box in box_array:
        center_array.append([(box[0]+box[2])//2, (box[1]+box[3])//2])
    
    return np.array(center_array)


def visualize_results(image, detection_results, draw=True):
    """visualize detection and classification results

    Args:
        image (array): test image
        detection_results (array): boxes 
        draw (bool, optional): whether to show the results. Defaults to True.

    Returns: 
        array: the result image
    """
    v= Visualizer(image[:, :, ::-1],
                        metadata=MetadataCatalog.get("tufts"), 
                        scale=0.5
            )
    out = v.draw_instance_predictions(detection_results)
    if draw:
        cv2.imshow("result",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    return out.get_image()[:, :, ::-1]
    

if __name__=="__main__":
    dataset_dir="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/"
    annotation_dir=dataset_dir+"Annotations/"
    image_dir=dataset_dir+"JPEGImages/"

    save_image=True
    output_dir="./output/"
    test_image_id="DSC_2422"
    ref_image_id="DSC_2410"

    # build the category dict for visualization
    ref_xml=parse_xml(annotation_dir+ref_image_id+".xml")
    MetadataCatalog.get("tufts").set(thing_classes=list(ref_xml.keys()))

    # detect bounding boxes for test image
    test_image_path=image_dir+test_image_id+".JPG"
    detection_results=detect_tufts_one_image(test_image_path)
    print(detection_results)

    # propagate class information for the bounding boxes
    boxes=detection_results["instances"].to("cpu")
    center_array=calc_center_from_box(boxes.pred_boxes)
    predict_result=predict_classification(test_image_id,dataset_dir, ref_image_id, target_box_center=center_array, draw=False)
    for i in range(boxes.pred_classes.shape[0]):
        boxes.pred_classes[i]=predict_result[i]

    # draw the final detection results
    image=cv2.imread(test_image_path)
    if save_image:
        results=visualize_results(image, boxes, draw=False)
        save_image_dir=output_dir+"final_results/"
        os.makedirs(save_image_dir, exist_ok=True)
        cv2.imwrite(save_image_dir+test_image_id+".png",results)
        print("image saved to "+save_image_dir+test_image_id+".png")
    else:
        visualize_results(image, boxes, draw=True)
