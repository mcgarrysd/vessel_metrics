#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:50:34 2022

Murine_segmentation 

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

data_path = '/media/sean/SP PHD U3/from_home/murine_data/sean_seg/'

label_list = []
im_list = []
data_files = os.listdir(data_path)
files = [f for f in data_files if not '.' in f]

for file in files:
    label = cv2.imread(data_path+file+'/label.png',0)
    label[label>0] =1
    label_list.append(label)
    im_list.append(cv2.imread(data_path+file+'/img.png',0))
    
seg_list = []
conn_list = []
area_list = []
length_list = []
jacc_list = []
Q_list = []
sigma1 = range(1,5,1); sigma2 = range(10,20,5)
for im, label in zip(im_list, label_list):
    seg = vm.multi_scale_seg(im, sigma1 = sigma1, sigma2 = sigma2, filter = 'meijering', thresh = 40, ditzle_size = 0, hole_size = 200)
    length, area, conn, Q = vm.cal(label.astype(np.uint8), seg.astype(np.uint8))
    seg_list.append(seg)
    conn_list.append(conn)
    area_list.append(area)
    length_list.append(length)
    jacc_list.append(vm.jaccard(label, seg))
    Q_list.append(Q)
    
overlay1 = label_list[0]*2+seg_list[0]
overlay2 = label_list[1]*2+seg_list[1]

vm.overlay_segmentation(im_list[0], overlay1)
vm.overlay_segmentation(im_list[1], overlay2)