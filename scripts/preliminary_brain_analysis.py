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
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from skan import draw


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



def contrast_stretch(image,upper_lim = 255, lower_lim = 0):
    c = np.percentile(image,5)
    d = np.percentile(image,95)
    
    stretch = (image-c)*((upper_lim-lower_lim)/(d-c))+lower_lim
    stretch[stretch<lower_lim] = lower_lim
    stretch[stretch>upper_lim] = upper_lim
    
    return stretch


seg = contrast_stretch(nr_no_blur)
seg = cv2.medianBlur(seg.astype(np.uint8),5)
plt.figure(); plt.imshow(seg)

####### alt preproc
alt = volume[30]
alt = contrast_stretch(alt)
alt = vm.preprocess_seg(alt)

plt.figure(); plt.imshow(alt)

_, seg_im = cv2.threshold(alt.astype(np.uint16), 100, 255, cv2.THRESH_BINARY)

plt.figure(); plt.imshow(seg_im)

######## contrast stretch 2

def contrast_stretch2(image,upper_lim = 255, lower_lim = 0):
    im_vec = np.reshape(image,-1)
    im_vec_reduced = im_vec[np.where(im_vec>0)]
    
    c = np.percentile(im_vec_reduced,5)
    d = np.percentile(im_vec_reduced,95)
    
    stretch = (image-c)*((upper_lim-lower_lim)/(d-c))+lower_lim
    stretch[stretch<lower_lim] = lower_lim
    stretch[stretch>upper_lim] = upper_lim
    
    return stretch

stretch2 = contrast_stretch2(no_reslice)
stretch2 = vm.preprocess_seg(stretch2)

plt.figure(); plt.imshow(stretch2)

_, seg_im2 = cv2.threshold(alt.astype(np.uint16), 100, 255, cv2.THRESH_BINARY)

plt.figure(); plt.imshow(seg_im2)

show_images = True
if show_images:
    plt.figure();
    plt.imshow(im)
    plt.figure();
    plt.imshow(no_reslice)
    plt.figure()
    plt.imshow(nr_no_blur)
    
for i in range(0,50,5):
    plt.figure()
    plt.imshow(volume[i])
    
reslice10 = vm.sliding_window(volume,10)
plt.figure(); plt.imshow(reslice10[10])

reslice15 = vm.sliding_window(volume,15)
plt.figure(); plt.imshow(reslice_15[10])

reslice10_nw = vm.reslice_image(volume,10)
plt.figure(); plt.imshow(reslice10_nw[0])

mip_slice = reslice10_nw[0]
plt.imshow(mip_slice)

mip_slice_preproc = vm.preprocess_seg(mip_slice)
plt.figure(); plt.imshow(mip_slice_preproc)

_, seg = cv2.threshold(mip_slice_preproc.astype(np.uint8),15,255, cv2.THRESH_BINARY)

plt.figure(); plt.imshow(seg)

fullseg = vm.segment_vessels(mip_slice, bin_thresh = 10)

# tubular filters test
sato_im = sato(mip_slice_preproc, sigmas = range(1,10,2), mode = 'reflect', black_ridges = False)
plt.figure(); plt.imshow(sato_im)

sato_max = np.max(sato_im)
sato_norm = sato_im/sato_max*255
sato_norm = np.round(sato_norm).astype(np.uint8)
plt.imshow(sato_norm)

sato_label = np.zeros_like(sato_norm)
sato_label[sato_norm>15] =1

overlap = sato_label + fullseg
plt.figure(); plt.imshow(overlap)

new_label = np.zeros_like(fullseg)
new_label[overlap>1] =1

# Testing vessel metrics on segmentation 
skel = skeletonize(new_label)
plt.figure(); 
skel_overlay = draw.overlay_skeleton_2d(mip_slice_preproc,skel,dilate = 2)

edges, bp = vm.find_branchpoints(skel)
_, l_edge_labels = cv2.connectedComponents(l_edges, connectivity = 8)
l_edge_labels, l_edges = vm.remove_small_segments(l_edge_labels, 50)
