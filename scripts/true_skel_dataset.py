#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:42:02 2021

true skel datasets

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

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/4dpf-5WT-7MM/'
out_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/true_skel/'
files = ['flk gata 4dpf Oct 5 2019 E1.czi', 'flk gata 4dpf Oct 5 2019 E2.czi', 'flk gata 4dpf Oct 5 2019 E3.czi', 'flk gata 4dpf Oct 5 2019 E4.czi']

count = 0
for f in files:
    vol = vm.preprocess_czi(data_path,f)
    vol = vm.sliding_window(vol, 4)

    slice_number = np.round(vol.shape[0]/2)
    
    slices = np.uint8(np.array([slice_number-3, slice_number, slice_number+3]))
    for s in slices:
        im = vol[s,:,:]
        im_name = 'fish' + str(count) + '_slice' + str(s) + '.png'
        cv2.imwrite(out_path+im_name, im)
    del vol
    gc.collect()
    count+=1
