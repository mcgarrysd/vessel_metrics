#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:24:42 2021

Alternative segmentation methods

@author: sean
"""

import skimage.segmentation as seg
import cv2
import numpy as np
import vessel_metrics as vm
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import regionprops
from skimage.measure import regionprops_table
from copy import deepcopy
from sklearn.cluster import KMeans
from cv2_rolling_ball import subtract_background_rolling_ball
import os

data_path = '/home/sean/Documents/segmentation/'

data_list = []
for file in os.listdir(data_path):
    d = os.path.join(data_path,file)
    if os.path.isdir(d) and 'fish' in d:
        data_list.append(d)

all_jacc = []
all_seg = []
for r in range(50):
    cv2.setRNGSeed(r)
    jacc_temp = []
    seg_temp = []
    for i in range(0,len(data_list),5):
        d = data_list[i]
        im = cv2.imread(d+'/img.png',0)
        label = cv2.imread(d+'/label.png',0)
        
        im = vm.clahe(im)
        seg = vm.segment_vessels(im)
        
        seg_temp.append(seg_temp)
        jacc_temp.append(vm.jaccard(label,seg))
    all_jacc.append(jacc_temp)
    all_seg.append(seg_temp)





