#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:10:49 2022

Pericyte_figure_ACHRI_2022

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

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/raw_data/DMH410uM75hpf/Jan15/'
data_list = os.listdir(data_path)
this_file = data_list[3]
volume = vm.preprocess_czi(data_path,this_file, channel = 1)
slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
pericyte_raw = reslice[0]
    
volume = vm.preprocess_czi(data_path,this_file, channel = 0)
slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
vessel_raw = reslice[0]

vessel_seg = vm.brain_seg(vessel_raw, filter = 'frangi', thresh = 10)
vessel_preproc = vm.preprocess_seg(vessel_raw)
vessel_preproc = vm.contrast_stretch(vessel_preproc)

peri_top_stretch = vm.contrast_stretch(pericyte_raw)

def crop_ventral_brain(im):
    new_im = im[350:650,200:500]
    return  new_im

preproc_crop = crop_ventral_brain(vessel_preproc)
vessel_crop = crop_ventral_brain(vessel_raw)
seg_crop = crop_ventral_brain(vessel_seg)
peri_crop = crop_ventral_brain(pericyte_raw)

plt.figure(); plt.imshow(peri_crop)

high_vals = np.zeros_like(peri_crop)
high_vals[peri_crop>75] = 1
vm.overlay_segmentation(peri_crop,high_vals, alpha = 0.5)

peri_seg = np.zeros_like(peri_crop)
peri_seg[(high_vals>0) & (seg_crop>0)]=1

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
combined_seg[seg_crop>0] = 1
combined_seg[reduced_label>0]=100

skel, edges, bp = vm.skeletonize_vm(vessel_seg)
skel_crop = crop_ventral_brain(skel)
edges_crop = crop_ventral_brain(edges)
bp_crop = crop_ventral_brain(bp)

edges_overlay = np.zeros_like(peri_seg)
edges_overlay = seg_crop+edges_crop*2+bp_crop*3


########################################################
# Panels 

plt.figure(); 
plt.imshow(vessel_crop, 'gray')
plt.figure(); 
plt.imshow(peri_crop, 'gray')
vm.overlay_segmentation(vessel_crop, combined_seg, alpha = 0.7)
vm.overlay_segmentation(vessel_crop, edges_overlay)
