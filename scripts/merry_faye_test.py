#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:57:55 2022

Merry faye data test

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
from aicsimageio import AICSImage
import timeit

data_path = '/media/sean/SP PHD U3/from_home/merry_faye_data/nov17/' 


file_list = os.listdir(data_path+'raw/')

net_length = []
net_length_um = []
im_name_list = []
mkdir = True
for file in file_list:
    fname = file.split('.')
    fname = fname[0].split(' ')
    und = '_'
    prefix = fname[0]+und+fname[1]+und+fname[-2]+und+fname[-1]
    print(file, prefix)
    
    volume, dims = vm.preprocess_czi(data_path+'raw/',file, channel = 1)
    slice_range = len(volume)
    slice_thickness = np.round(slice_range/2).astype(np.uint8)
    reslice = vm.reslice_image(volume,slice_thickness)
    if mkdir == True:
        os.mkdir(data_path+'processed/'+prefix)
    for i in range(reslice.shape[0]):
        this_slice = reslice[i]
        seg = vm.segment_image(this_slice, thresh = 40)
        
        skel, edges, bp = vm.skeletonize_vm(seg)
        nl = vm.network_length(edges)
        net_length.append(nl)
        net_length_um.append(nl*dims[1])
        overlay = edges*100+seg*50
        
        suffix = '_slice'+str(i)+'.png'
        im_name_list.append(prefix+'_slice'+str(i))

        cv2.imwrite(data_path+'processed/'+prefix+'/img'+suffix,this_slice)
        cv2.imwrite(data_path+'processed/'+prefix+'/label'+suffix, overlay)