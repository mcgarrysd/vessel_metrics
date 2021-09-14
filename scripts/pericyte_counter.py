#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:44:20 2021

Pericyte counter

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from timeit import default_timer as timer
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from scipy import stats

#######################################################################
# Create pericyte projection

data_path = '/home/sean/Documents/suchit_feb_21/'
data_list = os.listdir(data_path)
c1_bottom_list= []
c1_top_list = []
c0_bottom_list= []
c0_top_list = []
for d in data_list:
    volume = vm.preprocess_czi(data_path,d, channel = 1)
    slice_range = len(volume)
    slice_thickness = np.round(slice_range/2).astype(np.uint8)
    reslice = vm.reslice_image(volume,slice_thickness)
    c1_bottom_list.append(reslice[0])
    c1_top_list.append(reslice[1])
    
    volume = vm.preprocess_czi(data_path,d, channel = 0)
    slice_range = len(volume)
    slice_thickness = np.round(slice_range/2).astype(np.uint8)
    reslice = vm.reslice_image(volume,slice_thickness)
    c0_bottom_list.append(reslice[0])
    c0_top_list.append(reslice[1])
    
plt.figure()
plt.subplot(2,2,1)
plt.imshow(c0_bottom_list[0])
plt.subplot(2,2,2)
plt.imshow(c0_top_list[0])
plt.subplot(2,2,3)
plt.imshow(c1_bottom_list[0])
plt.subplot(2,2,4)
plt.imshow(c1_top_list[0])


peri_top = c1_top_list[0]
peri_bottom= c1_bottom_list[0]
vessel_top = c0_top_list[0]
vessel_bottom = c0_bottom_list[0]

vessel_top_seg = vm.brain_seg(vessel_top, sato_thresh = 40)
vm.overlay_segmentation(vessel_top, vessel_top_seg, contrast_adjust=True)

def crop_ventral_brain(im):
    new_im = im[350:650,700:1000]
    return  new_im

peri_top_stretch = vm.contrast_stretch(peri_top)

vtop_crop = crop_ventral_brain(vessel_top)
vseg_crop = crop_ventral_brain(vessel_top_seg)
peri_crop = crop_ventral_brain(peri_top_stretch)

plt.imshow(peri_crop)

high_vals = np.zeros_like(peri_crop)
high_vals[peri_crop>75] = 1
vm.overlay_segmentation(peri_crop,high_vals, alpha = 0.5)

peri_seg = np.zeros_like(peri_crop)
peri_seg[(high_vals>0) & (vseg_crop>0)]=1

kernel = np.ones((3,3),np.uint8)
peri_seg = cv2.morphologyEx(peri_seg, cv2.MORPH_OPEN, kernel)

num_labels, labels = cv2.connectedComponents(peri_seg.astype(np.uint8))

unique_labels = np.array(np.nonzero(np.unique(labels))).flatten()

reduced_label = np.zeros_like(peri_seg)
for u in unique_labels:
    numel = len(np.argwhere(labels == u))
    if numel>15 and numel<500:
        reduced_label[labels == u] = 1
        
vm.overlay_segmentation(peri_crop, reduced_label, alpha = 0.9)

combined_seg = np.zeros_like(peri_seg)
combined_seg[vseg_crop>0] = 1
combined_seg[reduced_label>0]=100

vm.overlay_segmentation(vtop_crop, combined_seg, alpha = 0.3)
