'''
predict the bounding box for images in a directory and save the results
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


config_file_path="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/configs/haicu/faster_rcnn_resnet50_fpn.yaml"
weight_file_path="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/experiments/haicu/res50fpn/model_0039999.pth"
dataset_dir="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/"
annotation_dir=dataset_dir+"Annotations/"
image_dir=dataset_dir+"JPEGImages1/"
output_dir="./output/"
ref_image_id="DSC_2410"


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
    cfg.merge_from_file(config_file_path)
    cfg.MODEL.WEIGHTS = weight_file_path
    cfg.freeze()
    return cfg


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


def detect_tufts_one_dir(image_dir):
    """detect the bounding boxes for each image in image_dir

    Args:
        image_dir (str): test image dir

    Returns:
        [dict]: a dict that stores the detection results
    """
    import json
    image_name_list=os.listdir(image_dir)
    result_dict=dict.fromkeys(image_name_list)
    cfg = setup()
    model=DefaultPredictor(cfg)
    for image_name in image_name_list:
        # check if image is in right format
        if image_name[-4:] not in [".png", ".JPG", "jpg", "bmp"]:
            print("not support image format for {}".format(image_name))
            continue
        image=cv2.imread(image_dir+image_name)
        outputs=model(image)
        boxes=outputs["instances"].to("cpu")
        result_dict[image_name]=boxes

    return result_dict


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


def predict_tufts(boxes_dict):
    """propagate class information for the detected bounding boxes

    Args:
        boxes_dict (dict): detection results without class info

    Returns:
        [dict]: results with class info
    """
    for k in boxes_dict.keys():
        boxes=boxes_dict[k]
        center_array=calc_center_from_box(boxes.pred_boxes)
        predict_result=predict_classification(k[:-4],dataset_dir, ref_image_id, target_box_center=center_array, draw=False)
        for i in range(boxes.pred_classes.shape[0]):
            boxes_dict[k].pred_classes[i]=predict_result[i]
    return boxes_dict


def save_files(boxes_dict, image_dir, output_dir):
    """save final results

    Args:
        boxes_dict (dict): dict that stores the prediction results
        image_dir (str): test image directory
        output_dir (str): image directory to save prediction results
    """
    for k in boxes_dict.keys():
        image=cv2.imread(os.path.join(image_dir,k))
        
        results=visualize_results(image, boxes_dict[k], draw=False)
        save_image_dir=output_dir+"final_results/"
        os.makedirs(save_image_dir, exist_ok=True)
        cv2.imwrite(save_image_dir+k+".png",results)
        print("image saved to "+save_image_dir+k+".png")


if __name__=="__main__":
    ref_xml=parse_xml(annotation_dir+ref_image_id+".xml")
    MetadataCatalog.get("tufts").set(thing_classes=list(ref_xml.keys()))
    detection_results=detect_tufts_one_dir(image_dir)
    prediction_results=predict_tufts(detection_results)
    save_files(prediction_results, image_dir, output_dir)
