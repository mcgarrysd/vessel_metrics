#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 08:38:52 2021

Characterize jasper data

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
from copy import deepcopy
import os
from shutil import copyfile
import gc

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/czis'
out_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/sample_images'
data_list = os.listdir(data_path+f)
count = 0
for f in data_folders:
    f = data_folders[3]
    data_list = os.listdir(data_path+f)
    fish_number = 0
    this_prefix = new_prefix[count]
    for i in data_list:
        fish_number+=1
        vol = vm.preprocess_czi(data_path+f,i)
        vol = vm.sliding_window(vol, 4)
        if 'flk gata inx' in i:
            mutant_status = 'wt'
        else:
            mutant_status = 'mutant'
        z_dim.append(vol.shape[0])
        x_dim.append(vol.shape[1])
        y_dim.append(vol.shape[2])
        mutant_list.append(mutant_status)
        timepoint.append(this_prefix)
        
        slice_number = np.round(vol.shape[0]/2)
        
        slices = np.uint8(np.array([slice_number-3, slice_number, slice_number+3]))
        for s in slices:
            im = vol[s,:,:]
            im_name = this_prefix + '_fish' + str(fish_number) + '_' + mutant_status + '_slice' + str(s) + '.png'
            cv2.imwrite(data_path+'sample_images/'+im_name, im)
        new_filename = this_prefix + '_fish' + str(fish_number) + '_' + mutant_status + '.czi'
        copyfile(data_path+f+i,out_path + new_filename)
        del vol
        gc.collect()
    count+=1
    
