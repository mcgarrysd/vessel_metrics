#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 08:58:24 2021

Segmentation efficacy dataset

@author: sean
"""

import cv2
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
from copy import deepcopy
import os
import gc

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/czis/'
out_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/segmentation/'
data_list = os.listdir(data_path)

reduced_data_list = []
for this_entry in data_list:
    if '3dpf' in this_entry and 'wt' in this_entry:
        reduced_data_list.append(this_entry)

for czi in reduced_data_list:
    vol = vm.preprocess_czi(data_path,czi)
    vol = vm.sliding_window(vol, 4)
    
    slice_number = np.round(vol.shape[0]/2)
    slices = np.uint8(np.array([slice_number-4, slice_number-2, slice_number, slice_number+2, slice_number+4]))
    
    time, fish, status = czi.split('_')
    
    for s in slices:
        im = vol[s,:,:]
        im_name = fish + '_slice' + str(s) + '.png'
        cv2.imwrite(out_path + im_name, im)
    del vol
    gc.collect()