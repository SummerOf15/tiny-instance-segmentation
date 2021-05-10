"""
optical flow method to propagate the labels
"""

import numpy as np
import cv2 as cv


dataset="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/JPEGImages/"
annotation_dir="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/Annotations/"

img1 = cv.imread(dataset+'DSC_2410.JPG')
img1=cv.resize(img1,(576,384))
prvs = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(img1)
hsv[...,1] = 255

frame2=cv.imread(dataset+'DSC_2411.JPG')
frame2=cv.resize(frame2,(576,384))
next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 5, 25, 3, 5, 1.2, 0)
# 368,181 -> 312,220
w=276
h=136
w1=233
h1=165
print("x={}, y={}".format(h+flow[h,w,0],w+flow[h,w,1]))
print("x={}, y={}".format(h+flow[h,w,1],w+flow[h,w,0]))
print("x={}, y={}".format(h1+flow[h1,w1,0],w1+flow[h1,w1,1]))
print("x={}, y={}".format(h1+flow[h1,w1,1],w1+flow[h1,w1,0]))

mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
cv.imshow("frame1",img1)
cv.imshow("frame2",frame2)
cv.imshow('flow',bgr)
cv.waitKey(0)