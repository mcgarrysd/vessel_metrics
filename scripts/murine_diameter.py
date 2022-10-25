#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:12:51 2022

Murine diameter

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
sigma1 = range(1,5,1); sigma2 = range(10,20,5)
for im, label in zip(im_list, label_list):
    seg = vm.multi_scale_seg(im, sigma1 = sigma1, sigma2 = sigma2, filter = 'meijering', thresh = 40, ditzle_size = 0, hole_size = 200)
    length, area, conn, Q = vm.cal(label.astype(np.uint8), seg.astype(np.uint8))
    seg_list.append(seg)

skel_list = []
edge_list = []
for seg in seg_list:
    skel, edges, bp = vm.skeletonize_vm(seg)
    skel_list.append(skel)
    _, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    edge_list.append(edge_labels)
    
#for edges,im in zip(edge_list, im_list):
#    vm.overlay_segmentation(im, edges)

test_segs = [38,69,5,5,4]

viz_list = []
diam_list = []
mean_diam_list = []
for seg_num, edges, im, seg in zip(test_segs, edge_list, im_list, seg_list):
    im = vm.preprocess_seg(im)
    diam,mean_diam, viz = vm.visualize_vessel_diameter(edges, seg_num, seg, im)
    viz_list.append(viz)
    diam_list.append(diam)
    mean_diam_list.append(mean_diam)
    # vm.overlay_segmentation(im, viz, alpha = 1)