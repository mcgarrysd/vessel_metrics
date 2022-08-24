#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 12:18:57 2022

Check sarah annotations
uneven cropping on 2nd annotation - this script only works for 1 image

@author: sean
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from copy import deepcopy

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/sarah_annotate/'

data_files = os.listdir(data_path)
 
seg_list = []
conn_list = []
area_list = []
length_list = []
jacc_list = []
for f in data_files:
    label = cv2.imread(data_path+f+'/label.png',0)
    label_s_temp = cv2.imread(data_path+f+'/label_s.png',0)
    label_s = np.zeros_like(label_s_temp)
    label_s[label_s_temp<250] = 1
    im = cv2.imread(data_path+f+'/img.png',0)
    seg = vm.brain_seg(im, filter = 'frangi', thresh = 20)
    seg_list.append(seg.astype(np.uint8))
    

        
    
    length, area, conn = vm.cal(label.astype(np.uint8), seg.astype(np.uint8))
    length_s, area_s, conn_s = vm.cal(label.astype(np.uint8), seg.astype(np.uint8))
    jacc = vm.jaccard(label, seg)
    jacc2 = vm.jaccard(label_s, seg)
    jacc3 = vm.jaccard(label_s, label)
    
    plt.figure(); plt.imshow(im)
    plt.figure(); plt.imshow(label)
    plt.figure(); plt.imshow(label_s)