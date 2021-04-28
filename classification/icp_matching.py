"""
use icp method to register the tufts position
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist


def icp(model, data, maxIter, thres, dist_thres, percent):
    """
    ICP (iterative closest point) algorithm
    Simple ICP implementation for teaching purpose
    - input
    model : scan taken as the reference position
    data : scan to align on the model
    maxIter : maximum number of ICP iterations
    thres : threshold to stop ICP when correction is smaller
    - output
    R : rotation matrix
    t : translation vector
    meandist : mean point distance after convergence
    """

    print('Running ICP, ', end='')

    # Various inits
    olddist = float("inf")  # residual error

    # Create array of x and y coordinates of valid readings for reference scan
    ref = np.array(model)

    # Create array of x and y coordinates of valid readings for processed scan
    dat = np.array(data)
    # dat=np.vstack((dat, dat[:2,:]))

    # Initialize transformation to identity
    R = np.eye(2)
    t = np.zeros((2, 1))

    meandist=0
    # Main ICP loop
    for iter in range(maxIter):
        ## KNN
        # ----- Find nearest Neighbors for each point, using kd-trees for speed
        tree = KDTree(ref)
        distance, index = tree.query(dat)
        meandist = np.mean(distance)
        dat_matched = []
        new_index = []
        index_set = set(index)
        for i in index_set:
            ind = np.where(index == i)[0]
            min_dis_ind = np.argmin(distance[ind])
            temp_dat = dat[ind[min_dis_ind],:]
            dat_matched.append(temp_dat)
            new_index.append(i)
        
        dat_matched = np.array(dat_matched)
        index = np.array(new_index)
        
        x = ref[index,:] - dat_matched
        temp=x.T@x # 2x2
        meandist=(temp[0,0]+temp[1,1])/2

        # Compute point mean
        mdat = np.mean(dat_matched, 0) #1x2
        mref = np.mean(ref[index, :], 0) #1x2

        # Use SVD for transform computation
        C = np.transpose(dat_matched-mdat) @ (ref[index, :] - mref)
        u, s, vh = np.linalg.svd(C)
        Ri = vh.T @ u.T
        Ti = mref - Ri @ mdat

        # Apply transformation to points
        dat = Ri @ dat.T
        dat = dat.T + Ti

        # Update global transformation
        R = Ri @ R
        t = Ri @ t + Ti.reshape(2, 1)

        # Stop when no more progress
        if abs(olddist-meandist) < thres:
            break

        # store mean residual error to check progress
        olddist = meandist

    print("finished with mean point corresp. error {:f}".format(meandist))

    return R, t, meandist



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
    

def main():

    dataset_dir="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/Annotations/"
    xml_files=os.listdir(dataset_dir)
    refer_file=xml_files[0]
    print("refered xml file is {}".format(refer_file))
    raw_file=xml_files[1]
    print("raw xml file is {}".format(raw_file))

    refer_dict=parse_xml(dataset_dir+refer_file)
    raw_dict=parse_xml(dataset_dir+raw_file)

    refer_array=np.array(list(refer_dict.values()))
    raw_array=np.array(list(raw_dict.values()))

    # Init displays
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(14, 7))

    c = np.random.rand(3,)
    ax1.scatter(refer_array[:,0], refer_array[:,1], color=c, s=1)
    # ax1.axis([-5.5, 12.5, -12.5, 6.5])
    ax1.set_title('reference')
    ax2.scatter(raw_array[:,0], raw_array[:,1], color=c, s=1)
    # ax1.axis([-5.5, 12.5, -12.5, 6.5])
    ax2.set_title('raw')
    plt.pause(0.1)


    R, t, error = icp(refer_array, raw_array, 200, 1e-7,0.4,0.85)

    # correct raw scans
    raw_array = np.matmul(R, np.transpose(raw_array)) + t

    # Display
    c = np.random.rand(3,)
    ax1.scatter(refer_array[:,0], refer_array[:,1], color=c, s=1)
    # ax1.axis([-5.5, 12.5, -12.5, 6.5])
    ax1.set_title('reference')

    ax2.scatter(raw_array[:,0], raw_array[:,1], color=c, s=1)
    # ax1.axis([-5.5, 12.5, -12.5, 6.5])
    ax2.set_title('raw')
    plt.pause(0.1)

    # plt.savefig('ICPLocalization.png')
    print("Press Q in figure to finish...")
    plt.show()

if __name__=="__main__":
    main()

