#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:32:15 2022

Vessel seg otsu

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

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/E1_segmentation/'

label_list = []
im_list = []
data_files = os.listdir(data_path)

for file in data_files:
    label_list.append(cv2.imread(data_path+file+'/label.png',0))
    im_list.append(cv2.imread(data_path+file+'/img.png',0))
    
seg_list = []
conn_list = []
area_list = []
length_list = []
jacc_list = []
Q_list = []
for im, label in zip(im_list, label_list):
    seg = vm.brain_seg(im, filter = 'meijering', thresh = 40)
    
    length, area, conn, Q = vm.cal(label.astype(np.uint8), seg.astype(np.uint8))
    conn_list.append(conn)
    area_list.append(area)
    length_list.append(length)
    jacc_list.append(vm.jaccard(label, seg))
    Q_list.append(Q)
    seg_list.append(seg)
  
seg_list_o = []
conn_list_o = []
area_list_o = []
length_list_o = []
jacc_list_o = []
Q_list_o = []
for im, label in zip(im_list, label_list):
    seg = otsu_seg(im, filter = 'meijering')
    
    length, area, conn, Q = vm.cal(label.astype(np.uint8), seg.astype(np.uint8))
    conn_list_o.append(conn)
    area_list_o.append(area)
    length_list_o.append(length)
    jacc_list_o.append(vm.jaccard(label, seg))
    Q_list_o.append(Q)
    seg_list_o.append(seg)


def otsu_seg(im, filter = 'meijering', sigmas = range(1,8,1), hole_size = 50, ditzle_size = 500, preprocess = True):
    if preprocess == True:
        im = vm.preprocess_seg(im)
    
    if filter == 'meijering':
        enhanced_im = meijering(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'sato':
        enhanced_im = sato(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'frangi':
        enhanced_im = frangi(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'jerman':
        enhanced_im = vm.jerman(im, sigmas = sigmas, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')
    norm = np.round(enhanced_im/np.max(enhanced_im)*255).astype(np.uint8)
    
    
    th, label = cv2.threshold(norm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((6,6),np.uint8)
    label = cv2.morphologyEx(label.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    _, label = vm.fill_holes(label.astype(np.uint8),hole_size)
    label = vm.remove_small_objects(label,ditzle_size)
    
    return label
