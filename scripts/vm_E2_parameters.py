#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:16:23 2021

vm_E2_parameters

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os

generate_data = True
if generate_data:
    data_path = '/home/sean/Documents/Data/suchit_mt/'
    data_list = glob.glob(data_path+'*.czi')
    out_path = '/home/sean/Documents/vm_manuscript/E2_parameters/'
    
    for i in data_list:
        vol = vm.preprocess_czi(i,"", channel = 1)
        slice_range = len(vol)
        slice_thickness = np.round(slice_range/2).astype(np.uint8)
        reslice = vm.reslice_image(vol,slice_thickness)
        this_slice = reslice[0]
        crop_slice = vm.crop_brain_im(this_slice)
        
        fish_name = i.split('/')[-1]
        emb_name = fish_name.split('_')[0]
        file_name = out_path+emb_name+'.png'
    
        cv2.imwrite(file_name,crop_slice)   
        