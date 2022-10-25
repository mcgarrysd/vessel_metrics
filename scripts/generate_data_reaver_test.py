#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:35:43 2022

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

data_path1 = '/media/sean/SP PHD U3/from_home/vm_manuscript/raw_data/suchit_feb_21/'
output_path = '/home/sean/Documents/from_home/vm_manuscript/reaver_test_data/all_ims_preproc/'

data_files = os.listdir(data_path1)

show_volumes = False 
reslice_list = []
for file in data_files:
    volume = vm.preprocess_czi(data_path1,file)
    slice_range = len(volume)
    slice_thickness = np.round(slice_range/2).astype(np.uint8)
    reslice = vm.reslice_image(volume,slice_thickness)
    this_slice = reslice[0].astype(np.uint8)
    this_slice = vm.preprocess_seg(this_slice)
    crop_slice = vm.crop_brain_im(this_slice)
    reslice_list.append(crop_slice)
if show_volumes:
    for i in reslice_list:
        plt.figure(); plt.imshow(i[0])
    
save_projections = True
if save_projections:
    for file, im in zip(data_files, reslice_list):
        fish = file.split('_')[0]
        im_name = fish+'.png'
        cv2.imwrite(output_path+im_name, im)
        
        
############################################################
        
        
data_path2 = '/media/sean/SP PHD U3/from_home/vm_manuscript/raw_data/Wnt Treatment/'

groups = ['Nov2', 'Nov14', 'Oct6']
for g in groups:
    data_files = os.listdir(data_path2+g+'/')
    data_files = [i for i in data_files if 'DMSO' in i]
    reslice_list = []
    for file in data_files:
        if g == 'Nov2' or g == 'Nov14' or g == 'Oct6':
            volume = vm.preprocess_czi(data_path2+g+'/',file, channel = 1)
            print('channel 1')
        else:
            volume = vm.preprocess_czi(data_path2+g+'/',file, channel = 0)
        slice_range = len(volume)
        slice_thickness = np.round(slice_range/2).astype(np.uint8)
        reslice = vm.reslice_image(volume,slice_thickness)
        this_slice = reslice[0]
        this_slice = vm.preprocess_seg(this_slice.astype(np.uint8))
        reslice_list.append(this_slice)
    if show_volumes:
        for i in reslice_list:
            plt.figure(); plt.imshow(i)
            
    save_projections = True
    if save_projections:
        for file, im in zip(data_files, reslice_list):
            fish = file.split(' ')[-1]
            fish = fish.split('.')[0]
            folder = g+'_'+fish
            im_name = folder+'.png'
            cv2.imwrite(output_path+im_name, im)
            

        