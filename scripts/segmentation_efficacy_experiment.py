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
    image = image.astype(np.uint16)
    
    clahe = cv2.createCLAHE(clipLimit = 40, tileGridSize = (16,16))
    cl = clahe.apply(im)
    cl_norm = np.round(cl/np.max(cl)*255)
    
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
    
    