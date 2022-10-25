#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:24:11 2022

Time lapse data

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

data_path = '/media/sean/SP PHD U3/from_home/murine_data/time_lapse_adam/'

files = os.listdir(data_path)

im_list = []
for file in files:
    vol = cv2.imreadmulti(data_path+file)
    im_list.append(vol[1])


test_vol = im_list[1]
vm.show_im(test_vol[5])


im1 = test_vol[5]
im2 = test_vol[500]

seg1 = vm.multi_scale_seg(im1)
vm.overlay_segmentation(im1,seg1)

seg2 = vm.multi_scale_seg(im2)
vm.overlay_segmentation(im2, seg2)







def multi_scale_seg(im, filter = 'meijering', sigma1 = range(1,8,1), sigma2 = range(10,20,5), hole_size = 50, ditzle_size = 500, thresh = 40, preprocess = True):
    if preprocess == True:
        im = preprocess_seg(im)
    
    if filter == 'meijering':
        enh_sig1 = meijering(im, sigmas = sigma1, mode = 'reflect', black_ridges = False)
        enh_sig2 = meijering(im, sigmas = sigma2, mode = 'reflect', black_ridges = False)
    elif filter == 'sato':
        enhanced_im = sato(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'frangi':
        enh_sig1 = frangi(im, sigmas = sigma1, mode = 'reflect', black_ridges = False)
        enh_sig2 = frangi(im, sigmas = sigma2, mode = 'reflect', black_ridges = False)
    elif filter == 'jerman':
        enh_sig1 = jerman(im, sigmas = sigma1, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')
        enh_sig2 = jerman(im, sigmas = sigma1, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')
    sig1_norm = normalize_contrast(enh_sig1)
    sig2_norm = normalize_contrast(enh_sig2)
    
    norm = sig1_norm.astype(np.uint16)+sig2_norm.astype(np.uint16)
    enhanced_label = np.zeros_like(norm)
    enhanced_label[norm>thresh] =1
    
    
    kernel = np.ones((6,6),np.uint8)
    label = cv2.morphologyEx(enhanced_label.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    _, label = fill_holes(label.astype(np.uint8),hole_size)
    label = remove_small_objects(label,ditzle_size)
    
    return label