#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:43:45 2022

human data test

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
import pandas as pd
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from scipy.stats import ttest_ind

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/raw_data/STARE/images/' 


test_im = cv2.imread(data_path+'im0001.ppm',0)

test_label = cv2.imread('/media/sean/SP PHD U3/from_home/vm_manuscript/raw_data/STARE/labels/im0001.ah.ppm',0)

sigmas = range(1,8,1)

##########################################################
enhanced_im2 = frangi(preproc, sigmas = sigmas, mode = 'reflect', black_ridges = True)
norm = np.round(enhanced_im2/np.max(enhanced_im2)*255).astype(np.uint8)
thresh = 10
enhanced_label = np.zeros_like(norm)
enhanced_label[norm>thresh] =1


kernel = np.ones((6,6),np.uint8)
label = cv2.morphologyEx(enhanced_label.astype(np.uint8), cv2.MORPH_OPEN, kernel)

_, label = fill_holes(label.astype(np.uint8),hole_size)
label = remove_small_objects(label,ditzle_size)

im2 = cv2.imread('/home/sean/Pictures/ordan_control1.png',0)

plt.figure(); plt.imshow(enhanced_im2)


###############################################################

raw_demean = vm.subtract_local_mean(test_im,8)
plt.figure(); plt.imshow(raw_demean)

inv_im = invert_im(test_im)
plt.figure(); plt.imshow(inv_im)

preproc2 = vm.preprocess_seg(inv_im.astype(np.uint8))
vm.show_im(preproc2)
inv_demean2 = vm.subtract_local_mean(preproc2)

test1 = vm.seg_no_thresh(inv_demean2, filter = 'jerman', sigmas = range(1,10,1), preprocess = False)
vm.show_im(test1)
test_seg = np.zeros_like(test1)
inds = np.where(test1>5 and test1<150)
test_seg[(test1>5) & (test1<150)] = 1

inv_demean = vm.subtract_local_mean(inv_im, 8)
plt.figure(); plt.imshow(inv_demean)


preproc = vm.preprocess_seg(inv_demean)
plt.figure(); plt.imshow(preproc)

def invert_im(im):
    output = np.zeros_like(im)
    im = vm.normalize_contrast(im)
    output = 255-im
    return output


def subtract_local_mean_sm(im,size = 8, bright_bg = True):
    if bright_bg == True:
        im_pad = np.pad(im, (size, size), 'maximum')
    
    else:
        im_pad = np.pad(im, (size,size), 'minimum')
    
    step = np.floor(size/2).astype(np.uint8)
    output = np.zeros_like(im_pad).astype(np.float16)
    for x in range(size,im_pad.shape[0]-size):
        for y in range(size,im_pad.shape[1]-size):
            region_mean = np.mean(im_pad[x-step:x+step, y-step:y+step]).astype(np.uint8)
            output[x,y] = im_pad[x,y]-region_mean
    output[output<0]=0
    output = np.round(output).astype(np.uint8)
    output_no_pad = output[size:im_pad.shape[0]-size, size:im_pad.shape[1]-size]
    return output_no_pad


def segment_retinal_vessels(im, invert = False):
    if invert == True:
        im == invert_im(im)
    im = im.astype(np.int16)
    preproc = vm.preprocess_seg(im)
    demean = subtract_local_mean_sm(preproc, size = 8)
    almost_seg = vm.seg_no_thresh(demean, filter = 'meijering', sigmas = (1,5,1))
    vm.show_im(almost_seg)
    seg = np.zeros_like(almost_seg)
    seg[almost_seg>35]=1
    
    kernel = np.ones((6,6),np.uint8)
    seg = cv2.morphologyEx(seg.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    seg = remove_small_objects(seg,500)
    
sizes = [4,8,16,32]
for s in sizes:
    demean = subtract_local_mean_sm(preproc, size = s)
    vm.show_im(demean)
    
def remove_background(im, kernel_size = 30):
    im_norm = vm.normalize_contrast(im)
    median_im = cv2.medianBlur(im, kernel_size)
    background = im/median_im
    bg_norm =vm.normalize_contrast(background)
    output = im_norm-bg_norm
    return output
    
vm.show_im(im_norm)
vm.show_im(median_im)
vm.show_im(bg_norm)
vm.show_im(output)

preproc = vm.preprocess_seg(output.astype(np.uint8))
almost_seg = vm.seg_no_thresh(preproc.astype(np.uint8), filter = 'meijering', sigmas = (1,5,1))

vm.show_im(preproc)
vm.show_im(almost_seg)
    
seg = np.zeros_like(almost_seg)
seg[almost_seg>10]=1

kernel = np.ones((6,6),np.uint8)
seg = cv2.morphologyEx(seg.astype(np.uint8), cv2.MORPH_OPEN, kernel)
seg = remove_small_objects(seg,500)

vm.show_im(seg)

test = cv2.bilateralFilter(im, 10, 75, 75)
