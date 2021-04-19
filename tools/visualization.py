"""visualize the detection results
"""

import json
from collections import OrderedDict
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# read prediction results
with open('/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/output/inference/tuft_test/coco_instances_results.json') as f:
    results_dict = json.load(f,object_pairs_hook=OrderedDict)

# read test dataset info
with open('/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/output/inference/tuft_test/tuft_test_coco_format.json') as f:
    dataset_dicts = json.load(f,object_pairs_hook=OrderedDict)

MetadataCatalog.get("tufts").set(thing_classes=["tufts"])
# for d in dataset_dicts["images"][0]:    
d=dataset_dicts["images"][0]
im = cv2.imread(d["file_name"])

v = Visualizer(im[:, :, ::-1],
                metadata=MetadataCatalog.get("tufts"), 
                scale=0.5, 
                )

for outputs in results_dict:
    if outputs["image_id"]=="DSC_2511": # visulize one image
        outputs["bbox"][2]=outputs["bbox"][0]+outputs["bbox"][2]
        outputs["bbox"][3]=outputs["bbox"][1]+outputs["bbox"][3]
        out = v.draw_box(outputs["bbox"])
    else:
        break
cv2.imshow("results",out.get_image()[:, :, ::-1])
cv2.waitKey(0)