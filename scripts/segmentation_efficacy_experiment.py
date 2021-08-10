#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:08:53 2021

segmentation efficacy

@author: sean
"""
import cv2
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
from copy import deepcopy
import os
import gc

image_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/segmentation/images/'
label_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/segmentation/labels/'

jacc = []; img_list = []; label_list = []; seg_list = []
conn_list = []; area_list = []; length_list = [];

data_list = os.listdir(label_path)
data_list = sort(data_list)
train_list = data_list[0:15]
test_list = data_list[15::]


for file in train_list:
    image = cv2.imread(image_path+file,0)
    label = cv2.imread(label_path+file,0)
    
    image = vm.clahe(image)
    
    label_list.append(label)
    seg = vm.segment_vessels(image, bin_thresh = 5)
    seg_list.append(seg)
    
    label_binary = np.uint8(label)
    seg_binary = np.uint8(seg)
    
    connectivity, area, length = vm.cal(label_binary,seg_binary)
    conn_list.append(connectivity)
    area_list.append(area)
    length_list.append(length)
    jacc.append(vm.jaccard(label_binary, seg_binary))
    
    print(file +' completed')
    
bad_inds = []
good_inds = []
for i in range(len(jacc)):
    if jacc[i]<0.6:
        bad_inds.append(i)
    else:
        good_inds.append(i)
        
# Best threshold
jacc_b = []; img_list = []; label_list = []; seg_list_b = []
        
bin_thresh = [1,3,5,7,10,12]
for file in train_list:
    image = cv2.imread(image_path+file,0)
    label = cv2.imread(label_path+file,0)
    
    image = vm.clahe(image)
    
    img_list.append(image)
    label_list.append(label)
    
    label_binary = np.uint8(label)
    
    
    seg_temp = []
    jacc_temp = []
    for b in bin_thresh:
        seg = vm.segment_vessels(image, bin_thresh = b)
        seg_temp.append(seg)
        seg_binary = np.uint8(seg)
        jacc_temp.append(vm.jaccard(label_binary,seg_binary))
    
    seg_list_b.append(seg_temp)
    jacc_b.append(jacc_temp)
    
    print(file +' completed')    