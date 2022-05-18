#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:30:28 2021
lab meeting sandbox 10/18/21

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from timeit import default_timer as timer
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from scipy import stats

wt_path = '/home/sean/Documents/suchit_wt_projections/'
wt_names = ['emb2', 'emb6', 'emb8', 'emb9']
wt_ims = []
wt_seg = []
for im_name in wt_names:
    im = cv2.imread(wt_path+im_name+'.png',0)
    wt_ims.append(im)
    wt_seg.append(vm.brain_seg(im))
    
    
    
#####################################################################
# Bright and Dim segmentation
bright_im = wt_ims[0]
bright_seg = wt_seg[0]

dim_im = wt_ims[3]
dim_seg = wt_seg[3]

contrast_dim = vm.contrast_stretch(dim_im)
plt.figure(); plt.imshow(contrast_dim,'gray')

contrast_bright= vm.contrast_stretch(bright_im)
plt.figure(); plt.imshow(contrast_bright,'gray')

bright_preproc = vm.preprocess_seg(contrast_bright)
dim_preproc = vm.preprocess_seg(contrast_dim)

plt.figure(); plt.imshow(bright_preproc,'gray')
plt.figure(); plt.imshow(dim_preproc,'gray')

bright_sato = sato(bright_preproc, sigmas = range(1,10,2), mode = 'reflect', black_ridges = False)
dim_sato = sato(dim_preproc, sigmas = range(1,10,2), mode = 'reflect', black_ridges = False)

plt.figure(); plt.imshow(bright_sato, 'gray')
plt.figure(); plt.imshow(dim_sato,'gray')

vm.overlay_segmentation(bright_im, bright_seg, alpha = 0.5)
vm.overlay_segmentation(dim_im, dim_seg, alpha = 0.5)