'''
evaluate the affine matching results
using matched points to calculate the transformation matrix and register the images,
then predict the location of corresponding points in target image
'''
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.kdtree import KDTree
import os
from align_transform import Align
from pycm import *
import matplotlib.pyplot as plt


dataset="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/JPEGImages/"
annotation_dir="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/Annotations/"
downsample_ratio=5

def parse_xml(xml_path):
    """parse xml file and calculate center point of each bounding box

    Args:
        xml_path (str): path of xml file

    Returns:
        str: dict variable includes center points and id of the bounding box
    """
    box_dict={}
    tree = ET.parse(xml_path)
    for obj in tree.findall("object"):
        cls = "tuft"
        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        center_point=[(bbox[0]+bbox[2])/2//downsample_ratio,(bbox[1]+bbox[3])/2//downsample_ratio] # x, y, downsample ratio=5
        box_dict.setdefault(obj.find("name").text, center_point)

    return box_dict


def predict(pred_image_name):
    """predict the location of tufts in pred_image

    Args:
        pred_image_name (str): id of target image

    Returns:
        array: the predicted tufts location [2xN]
    """
    ref_image_path=dataset+'DSC_2410.JPG' # source/reference image path
    ref_xml=parse_xml(annotation_dir+'DSC_2410.xml')  # reference image bounding box
    ref_xml_array=np.array(list(ref_xml.values()))
    
    target_image_path = dataset+pred_image_name+'.JPG'        # target image/image to be predicted
    
    # calculate the tranformation matrix and predict the corresponding points
    a=Align(ref_image_path,target_image_path,threshold=1)
    transformed_array=a.align_points(ref_xml_array.T)
    return list(ref_xml.keys()),transformed_array


def evaluate(pred_image_name):
    """evaluate the transformation performance

    Args:
        pred_image_name (str): the target image id

    Returns:
        list: annotated class
        list: predicted class 
    """
    ref_image_path=dataset+'DSC_2410.JPG' # source/reference image path
    ref_xml=parse_xml(annotation_dir+'DSC_2410.xml')  # reference image bounding box
    
    # check if the annotation file exists
    if not os.path.isfile(annotation_dir+pred_image_name+'.xml'):
        return None,None

    target_image_path = dataset+pred_image_name+'.JPG'        # target image/image to be predicted
    target_xml=parse_xml(annotation_dir+pred_image_name+'.xml')  # target image annotations, used for evaluation
    ref_xml_array=np.array(list(ref_xml.values()))
    target_xml_array=np.array(list(target_xml.values()))

    # calculate the tranformation matrix and predict the corresponding points
    a=Align(ref_image_path,target_image_path,threshold=1)
    transformed_array=a.align_points(ref_xml_array.T)

    ref_keys=list(ref_xml.keys())
    target_keys=list(target_xml.keys())

    box_tree2=KDTree(target_xml_array)  # a KD-tree to search for the nearest annotations

    correct_num=0 # correct prediction
    matching_dist=[] # matching distance 
    predict_keys=[] # predict result

    for i in range(transformed_array.shape[1]):
        dist1, ind=box_tree2.query(transformed_array[:,i].T,k=1) # choose the nearest point of predicted result
        predict_keys.append(target_keys[ind])
        if ref_keys[i]==target_keys[ind]: #check if the class info is the same
            correct_num+=1
            matching_dist.append(dist1) # store the distance for each pair of matching

    print("correct/total: {}/{}, rate: {}".format(correct_num,len(target_keys),correct_num/len(target_keys)))
    print("total tufts: xml1({}),xml2({})".format(len(ref_keys),len(target_keys)))
    
    return ref_keys, predict_keys


if __name__=="__main__":
    EVALUATION=True
    if EVALUATION:
        total_ref=[]
        total_predict=[]
        for i in range(1300):
            print("test DSC_{}".format(2411+i))
            ref_keys, predict_keys=evaluate("DSC_{}".format(2411+i))
            if ref_keys is None:
                continue
            total_ref+=ref_keys
            total_predict+=predict_keys
        
        print("--------final confusion matrix--------")
        cm = ConfusionMatrix(actual_vector=total_ref, predict_vector=total_predict)
        print("Accuracy: {}, Recall: {}, Precision:{}, F1: {}".format(cm.Overall_ACC, cm.TPR_Macro,cm.PPV_Macro, cm.F1_Macro))
        cm.plot(cmap=plt.cm.Greens,number_label=True,plot_lib="matplotlib")
        plt.show()
    else:
        image_name="DSC_2022"
        k,v=predict(image_name)
        for i in range(len(k)):
            print("tuft {}: {},{}".format(k[i],v[0,i],v[1,i]))

    
