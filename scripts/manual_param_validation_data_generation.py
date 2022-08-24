#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:13:39 2022

Manual param validation data generation

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
from scipy.stats import ttest_ind

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/raw_data/Wnt Treatment/Nov2/'

output_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/SE4_manual_params/'

data_files = os.listdir(data_path)
data_files = [i for i in data_files if 'DMSO' in i]

make_dir = True
for file in data_files:
    if make_dir:
        dir_name = file.split(' ')[-1]
        dir_name = dir_name.split('.')[0]
        mkdir(output_path+dir_name)
    volume = vm.preprocess_czi(data_path,file, channel = 1)
    slice_range = len(volume)
    slice_thickness = np.round(slice_range/2).astype(np.uint8)
    reslice = vm.reslice_image(volume,slice_thickness)
    this_slice = reslice[0]
    crop_slice_raw = crop_center(this_slice)
    this_slice = vm.preprocess_seg(this_slice.astype(np.uint8))
    crop_slice_proc = crop_center(this_slice)
    seg = vm.brain_seg(crop_slice_proc, filter = 'meijering', thresh = 40, preprocess = False)
    cv2.imwrite(output_path+dir_name+'/img.png', crop_slice_raw)
    cv2.imwrite(output_path+dir_name+'/label.png', seg)
    cv2.imwrite(output_path+dir_name+'/img_preproc.png', crop_slice_proc)
    
def crop_center(im):
    xdim = np.round(im.shape[0]/3).astype(np.uint16)
    ydim = np.round(im.shape[1]/3).astype(np.uint16)

    cropped_im = im[xdim:xdim*2,ydim:ydim*2]
    
    