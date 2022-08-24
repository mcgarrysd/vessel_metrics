#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:27:40 2022

Cynthia_network_length

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

data_path = '/media/sean/SP PHD U3/from_home/cynthia_network_length/may23/'

file_list = os.listdir(data_path+'Raw/')


net_length = []
prefix_list = []
mkdir = True
for file in file_list:
    fname = file.split('.')
    fname = fname[0].split(' ')
    und = '_'
    prefix = fname[0]+und+fname[1]+und+fname[-2]+und+fname[-1]
    prefix_list.append(prefix)
    
    volume = vm.preprocess_czi(data_path+'Raw/',file, channel = 0)
    slice_range = len(volume)
    slice_thickness = np.round(slice_range/2).astype(np.uint8)
    reslice = vm.reslice_image(volume,slice_thickness)
    this_slice = reslice[0]
    seg = vm.brain_seg(this_slice.astype(np.uint8), thresh = 40)
    
    skel, edges, bp = vm.skeletonize_vm(seg)
    net_length.append(vm.network_length(edges))
    overlay = edges*100+seg*50
    
    if mkdir == True:
        os.mkdir(data_path+'Processed/'+prefix)
    cv2.imwrite(data_path+'Processed/'+prefix+'/img.png',this_slice)
    cv2.imwrite(data_path+'Processed/'+prefix+'/label.png', overlay)
     
