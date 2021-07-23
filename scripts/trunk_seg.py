#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:44:21 2020

Segmentation of trunk depth image

@author: sean
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.close('all')

data_path ='/home/sean/Documents/Calgary_postdoc/Data/mCherryAug2020/'
img = cv2.imread(data_path + 'Fish2_trunk.tif',0)
ground_truth = cv2.imread(data_path + 'fish2_trunk/label.png',0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
cl1 = clahe.apply(img)

median = cv2.medianBlur(cl1, 11)

ret, otsu = cv2.threshold(median, 0,150, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, th1 = cv2.threshold(median, 15, 255, cv2.THRESH_BINARY)
adaptive = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,1)

titles = ['grayscale image', 'contrast equalization', 'Salt and pepper filter','Otsus threshold', 'Binary', 'adaptive']
images = [img, cl1, median, otsu, th1, adaptive]

for i in range(6):
    plt.subplot(3,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

adaptive2 = adaptive;
adaptive2[th1==0]=0;

connectivity = 8
num_labels, stat_labels, stats, centroids = cv2.connectedComponentsWithStats(adaptive2,connectivity, cv2.CV_32S)

new_mask = np.zeros(img.shape)

for i in range(num_labels-1):
    if stats[i+1,4] > 500:
        new_mask[stat_labels == i+1] = 1

plt.figure(2); plt.imshow(otsu); 
plt.figure(3); plt.imshow(th1); 
plt.figure(4); plt.imshow(img); 
plt.figure(5); plt.imshow(adaptive); 
plt.figure(6); plt.imshow(adaptive2);
plt.figure(7); plt.imshow(new_mask); 

plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
close = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel, iterations = 1)
plt.figure(8); plt.imshow(close)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,3))
close_horizontal = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel, iterations = 3)
plt.figure(9); plt.imshow(close_horizontal)


plt.figure(10); plt.imshow(img,"gray");
plt.imshow(close_horizontal, cmap = 'jet', alpha = 0.2)

plt.figure(11);
plt.subplot(1,3,1); plt.imshow(img,"gray"); plt.title("base image")
plt.subplot(1,3,2); plt.imshow(img,"gray"); plt.imshow(close_horizontal, cmap = 'jet', alpha = 0.2); plt.title("Seans segmentation")
plt.subplot(1,3,3); plt.imshow(img,"gray"); plt.imshow(ground_truth, cmap = 'jet', alpha = 0.2); plt.title("ground truth")

ground_truth = (ground_truth>1).astype(np.uint8)
overlap_mask = ground_truth + 2*close_horizontal
plt.figure(); plt.imshow(overlap_mask)

ground_truth_bool = ground_truth.astype(np.bool)
mask_bool = close_horizontal.astype(np.bool)
dice = np.sum(np.bitwise_and(ground_truth_bool, mask_bool))/np.sum(np.bitwise_or(ground_truth_bool,mask_bool))


#plt.figure(2)
#small_img = cv2.pyrDown(median)
#plt.hist(small_img)
#plt.show()