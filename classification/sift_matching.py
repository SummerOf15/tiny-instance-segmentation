"""
sift image matching then infer the position of tufts.
"""

import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.spatial.kdtree import KDTree
import os

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
        # We include "difficult" samples in training.
        # Based on limited experiments, they don't hurt accuracy.
        # difficult = int(obj.find("difficult").text)
        # if difficult == 1:
        # continue

        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        center_point=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2] # x, y
        box_dict.setdefault(obj.find("name").text, center_point)

    return box_dict

def main(pred_image_name):
    dataset="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/JPEGImages/"
    refer_dir="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/2019/"
    annotation_dir="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/Annotations/"

    img1 = cv.imread(dataset+'DSC_2410.JPG',cv.IMREAD_GRAYSCALE)          # queryImage
    xml1=parse_xml(annotation_dir+'DSC_2410.xml')  # image1 bounding box
    
    if not os.path.isfile(annotation_dir+pred_image_name+'.xml'):
        return 0,0
    img2 = cv.imread(dataset+pred_image_name+'.JPG',cv.IMREAD_GRAYSCALE)          # trainImage
    xml2=parse_xml(annotation_dir+pred_image_name+'.xml')  # image2 bounding box

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)  #query, train, return 2 matches
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # matched points list
    pt1_list=[] # in img1
    pt2_list=[] # in img2
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches): # m: best match, n: second match
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            pt1_list.append(list(kp1[m.queryIdx].pt))
            pt2_list.append(list(kp2[m.trainIdx].pt))

    pt1_array=np.array(pt1_list)
    pt2_array=np.array(pt2_list)
    # match_tree1=KDTree(pt1_array)
    match_tree2=KDTree(pt2_array)
    
    xml1_array=np.array(list(xml1.values()))
    xml2_array=np.array(list(xml2.values()))
    box_tree1=KDTree(xml1_array)  # box tree
    # box_tree2=KDTree(xml2_array)  # box tree
    keys1=list(xml1.keys())
    keys2=list(xml2.keys())

    match2_dict={key:["default",1000.0] for key in xml2} # here can be changed to 1,2,3,4,5... number of boxes
    correct_num=0
    for key in keys2:  #find the project of each box in test image
        dist1, ind=match_tree2.query(xml2[key],k=1) #find the closest matching point
        dist2, ind2=box_tree1.query(pt1_array[ind],k=1) # go to reference image and find the closest bounding box
        if key == keys1[ind2]:
            # print("CORRECT {}->{}, dist={}".format(key,keys1[ind2],dist2))
            correct_num+=1
        # else:
            # print("ERROR {}->{}, dist={}".format(key,keys1[ind2],dist2))
            
    print("correct/total: {}/{}, rate: {}".format(correct_num,len(keys2),correct_num/len(keys2)))
    print("total tufts: xml1({}),xml2({})".format(len(list(xml1.keys())),len(keys2)))
    
    return correct_num, len(keys2)
    # draw results
    # draw_params = dict(matchColor = (0,255,0),
    #                    singlePointColor = (255,0,0),
    #                    matchesMask = matchesMask,
    #                    flags = cv.DrawMatchesFlags_DEFAULT)
    # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    # plt.figure(figsize=(20,10))
    # plt.imshow(img3)
    # plt.show()

if __name__=="__main__":
    total_correct=0
    total_len=0
    for i in range(21):
        print("test DSC_{}".format(2411+i))
        c,t=main("DSC_{}".format(2411+i))
        total_correct+=c
        total_len+=t
    print("correct/total: {}/{}, rate: {}".format(total_correct,total_len,total_correct/total_len))