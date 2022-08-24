#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 12:20:50 2021

vasodilator test

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


data_path = '/home/sean/Documents/vasodilator/' 
output_path = '/home/sean/Documents/vm_manuscript/E4_vasodilator/'

data_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

show_volumes = False
preproc_data = True
if show_volumes:   
    reslice_list = []
    for file in data_files:
        volume = vm.preprocess_czi(data_path,file, channel = 1)
        slice_range = len(volume)
        slice_thickness = np.round(slice_range/2).astype(np.uint8)
        reslice = vm.reslice_image(volume,slice_thickness)
        this_slice = reslice[0]
        this_slice = vm.preprocess_seg(this_slice)
        reslice_list.append(this_slice)
    
save_projections = False
if save_projections:
    for file, im in zip(data_files, reslice_list):
        fish = file.split('_')[0]
        im_name = fish + '.png'
        cv2.imwrite(output_path +im_name, im)
        
        
#####################################################################
# same emb


data_path = '/home/sean/Documents/vasodilator/same_emb/' 
output_path = '/home/sean/Documents/vm_manuscript/E4_vasodilator/same_emb/'

data_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

show_volumes = False
preproc_data = True
if show_volumes:   
    reslice_list = []
    for file in data_files:
        volume = vm.preprocess_czi(data_path,file, channel = 1)
        slice_range = len(volume)
        slice_thickness = np.round(slice_range/2).astype(np.uint8)
        reslice = vm.reslice_image(volume,slice_range)
        this_slice = reslice[0]
        this_slice = vm.preprocess_seg(this_slice)
        reslice_list.append(this_slice)
    
save_projections = False
if save_projections:
    for file, im in zip(data_files, reslice_list):
        fish = file.split('.')[0]
        im_name = fish + '_full.png'
        cv2.imwrite(output_path +im_name, im)
        




