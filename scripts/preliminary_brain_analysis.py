#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 12:13:08 2021

Preliminary brain analysis

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from timeit import default_timer as timer
from cv2_rolling_ball import subtract_background_rolling_ball

plt.close('all')

data_path = '/home/sean/Documents/suchit_feb_21/'
file = 'emb1_Pdgfrbmc flkGFP 75 hpf.czi'

volume = vm.preprocess_czi(data_path, file)
reslice_volume = vm.sliding_window(volume, 4)

im = reslice_volume[30]
no_reslice = volume[30]

im_preproc = vm.preprocess_seg(im)
no_r_preproc = vm.preprocess_seg(no_reslice)

nr_no_blur, background = subtract_background_rolling_ball(no_reslice.astype(np.uint8), 200, light_background=False, use_paraboloid=False, do_presmooth=True)

plt.imshow(nr_no_blur)

def contrast_stretch(image,upper_lim = 255, lower_lim = 0):
    c = np.percentile(image,5)
    d = np.percentile(image,95)
    
    stretch = (image-c)*((upper_lim-lower_lim)/(d-c))+lower_lim
    stretch[stretch<lower_lim] = lower_lim
    stretch[stretch>upper_lim] = upper_lim
    
    return stretch

show_images = True
if show_images:
    plt.figure();
    plt.imshow(im)
    plt.figure();
    plt.imshow(no_reslice)