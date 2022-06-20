#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:27:39 2022

Vessel enhancement example
used for lab meeting presentation on 5/30/22

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

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/E1_combined/'

label_list = []
im_list = []
data_files = os.listdir(data_path)

file = data_files[16]
label = cv2.imread(data_path+file+'/label.png',0)
im = cv2.imread(data_path+file+'/img.png',0)

plt.figure(); plt.imshow(im, cmap = 'gray')

im_preproc = vm.preprocess_seg(im)

sigmas = range(1,10,2)
meij = meijering(im_preproc, sigmas = sigmas, mode = 'reflect', black_ridges = False)
s2 = range(1,20,2)
meij2 = meijering(im_preproc, sigmas = s2, mode = 'reflect', black_ridges = False)

plt.figure(); plt.imshow(meij)
plt.figure(); plt.imshow(meij2)

sato_im = sato(im_preproc, sigmas = sigmas, mode = 'reflect', black_ridges = False)
frangi_im = frangi(im_preproc, sigmas = sigmas, mode = 'reflect', black_ridges = False)
jerman_im = vm.jerman(im_preproc, sigmas = sigmas, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')

plt.figure(); plt.imshow(sato_im)
plt.figure(); plt.imshow(frangi_im)
plt.figure(); plt.imshow(jerman_im)

median_size = 7; ball_size = 400
im2 = vm.contrast_stretch(im)
#im2, background = subtract_background_rolling_ball(im2, ball_size, light_background=False,
                                                            use_paraboloid=False, do_presmooth=True)
im2 = cv2.medianBlur(im2.astype(np.uint8),median_size)
plt.figure(); plt.imshow(im2)

jerman2 = vm.jerman(im2, sigmas = sigmas, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')
