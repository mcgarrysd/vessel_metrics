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

data_path = '/home/sean/Documents/vessel_metrics/data/suchit_feb_21/'
data_list = os.listdir(data_path)
reslice_list_c1 = []
reslice_list_c0 = []
for d in data_list:
    volume = vm.preprocess_czi(data_path,d, channel = 1)
    slice_range = len(volume)
    slice_thickness = np.round(slice_range/2).astype(np.uint8)
    reslice = vm.reslice_image(volume,slice_thickness)
    reslice_list_c1.append(reslice[0])
    
    volume = vm.preprocess_czi(data_path,d, channel = 0)
    slice_range = len(volume)
    slice_thickness = np.round(slice_range/2).astype(np.uint8)
    reslice = vm.reslice_image(volume,slice_thickness)
    reslice_list_c0.append(reslice[0])
    
plt.figure();
plt.imshow(reslice_list_c0[0])
plt.figure(); 
plt.imshow(reslice_list_c1[0])
    



wt_path = '/home/sean/Documents/vessel_metrics/data/suchit_wt_projections/'
wt_names = ['emb3', 'emb6']
wt_ims = []
wt_seg = []
for im_name in wt_names:
    im = cv2.imread(wt_path+im_name+'.png',0)
    wt_ims.append(im)
    wt_seg.append(vm.brain_seg(im))
    
mt_path = '/home/sean/Documents/vessel_metrics/data/suchit_mt_projections/'
mt_names = ['emb3', 'emb15']
mt_ims = []
mt_labels = []
mt_seg = []
for im_name in mt_names:
    im = cv2.imread(mt_path+im_name+'.png',0)
    mt_ims.append(im)
    mt_labels.append(cv2.imread(mt_path+im_name+'/label.png',0))
    mt_seg.append(vm.brain_seg(im))