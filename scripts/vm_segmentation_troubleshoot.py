#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:50:19 2022

Segmentation trouble shooting
An attempt to figure out why the segmentation numbers are off

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

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/E1_combined/' 


label_list = []
im_list = []
data_files = os.listdir(data_path)

for file in data_files:
    temp_label = cv2.imread(data_path+file+'/label.png',0)
    temp_label[temp_label>0]=1
    label_list.append(temp_label)
    im_list.append(cv2.imread(data_path+file+'/img.png',0))
    
plt.close('all')
seg_list = []
conn_list = []
area_list = []
length_list = []
jacc_list = []
preproc_list = []
for im, label in zip(im_list, label_list):
    seg = brain_seg(im, filter = 'frangi', thresh = 30, ditzle_size = 2000, sigmas = range(1,8,1))
    seg_list.append(seg.astype(np.uint8))
    
    preproc_list.append(preprocess_seg(im))
    length, area, conn = vm.cal(label.astype(np.uint8), seg.astype(np.uint8))
    conn_list.append(conn)
    area_list.append(area)
    length_list.append(length)
    jacc_list.append(vm.jaccard(label, seg))
    
plt.figure(); plt.imshow(preproc_list[11])
vm.overlay_segmentation(im_list[11], seg_list[11]+label_list[11]*2)

plt.figure(); plt.imshow(preproc_list[17])
vm.overlay_segmentation(im_list[17], seg_list[17]+label_list[17]*2)

def contrast_stretch(image,upper_lim = 255, lower_lim = 0):
    c = np.percentile(image,5)
    d = np.percentile(image,95)
    
    stretch = (image-c)*((upper_lim-lower_lim)/(d-c))+lower_lim
    stretch[stretch<lower_lim] = lower_lim
    stretch[stretch>upper_lim] = upper_lim
    
    return stretch

def preprocess_seg(image,ball_size = 400, median_size = 7, upper_lim = 255, lower_lim = 0):
    image, background = subtract_background_rolling_ball(image, ball_size, light_background=False,
                                                            use_paraboloid=False, do_presmooth=True)
    image = cv2.medianBlur(image.astype(np.uint8),median_size)
    image = contrast_stretch(image, upper_lim = upper_lim, lower_lim = lower_lim)
    return image

def brain_seg(im, filter = 'meijering', sigmas = range(1,8,1), hole_size = 50, ditzle_size = 500, thresh = 60, preprocess = True):
    if preprocess == True:
        im = preprocess_seg(im)
    
    if filter == 'meijering':
        enhanced_im = meijering(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'sato':
        enhanced_im = sato(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'frangi':
        enhanced_im = frangi(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'jerman':
        enhanced_im = vm.jerman(im, sigmas = sigmas, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')
    norm = np.round(enhanced_im/np.max(enhanced_im)*255).astype(np.uint8)
    
    enhanced_label = np.zeros_like(norm)
    enhanced_label[norm>thresh] =1
    
    
    kernel = np.ones((6,6),np.uint8)
    label = cv2.morphologyEx(enhanced_label.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    _, label = fill_holes(label.astype(np.uint8),hole_size)
    label = remove_small_objects(label,ditzle_size)
    
    return label
    
    