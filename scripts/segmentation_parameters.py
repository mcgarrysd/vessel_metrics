#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:13:16 2021

segmentation parameter tuning

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from timeit import default_timer as timer

def segment_vessels(image,k = 12, hole_size = 250, ditzle_size = 750):
    image = cv2.medianBlur(image.astype(np.uint8),21)
    image, background = subtract_background_rolling_ball(image, 400, light_background=False,
                                         use_paraboloid=False, do_presmooth=True)
    im_vector = image.reshape((-1,)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(im_vector,k,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = center.astype(np.uint8)
    label_im = label.reshape((image.shape))
    seg_im = np.zeros_like(label_im)
    for i in np.unique(label_im):
        seg_im = np.where(label_im == i, center[i],seg_im)
    
    _, seg_im = cv2.threshold(seg_im.astype(np.uint16), 1, 255, cv2.THRESH_BINARY)
    
    _, seg_im = vm.fill_holes(seg_im.astype(np.uint8),hole_size)
    seg_im = vm.remove_small_objects(seg_im,ditzle_size)
    
    return seg_im


data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/annotation_samples/'

os.chdir(data_path)
file_list = []
for file in glob.glob("E*"):
    file_list.append(file)

jacc = []
start = timer()
for file in file_list:

    image = cv2.imread(data_path + file + '/img.png',0)
    label = cv2.imread(data_path + file + '/label.png',0)
    label[label>0] = 1
    
    test_label = segment_vessels(image, k = 12, hole_size = 500, ditzle_size = 250)
    overlay = test_label*2 + label
    
    jacc.append(vm.jaccard(label,test_label))

end = timer()
print(end-start)
jacc = np.asarray(jacc)
mean_jaccard = np.mean(jacc)
plt.figure(); plt.imshow(test_label)

# problem imagge
file = file_list[1]
image = cv2.imread(data_path + file + '/img.png',0)
label = cv2.imread(data_path + file + '/label.png',0)
label[label>0] = 1
    
test_label = segment_vessels(image, k = 12, hole_size = 500, ditzle_size = 750)
overlay = test_label*2 + label
jac_test = vm.jaccard(label,test_label)
