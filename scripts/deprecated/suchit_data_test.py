#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:20:23 2021

test script, to see if segmentation algorithm developed in tail images
functions on head images

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from timeit import default_timer as timer

plt.close('all')

data_path = '/home/sean/Documents/Calgary_postdoc/Data/suchit_feb_21/'
file = 'emb1_Pdgfrbmc flkGFP 75 hpf.czi'

volume = vm.preprocess_czi(data_path, file)
reslice_volume = vm.sliding_window(volume, 4)

#for i in range(23,28):
#    plt.figure()
#    plt.imshow(reslice_volume[i,:,:])
    
test_slice = reslice_volume[5,:,:]

test_label = vm.segment_vessels(test_slice)
test_label2 = vm.segment_vessels(test_slice, k = 16)
test_label2 = vm.segment_vessels(test_slice, k = 8)


def segment_vessels(image,k = 12, hole_size = 500, ditzle_size = 750):
    image = cv2.medianBlur(image.astype(np.uint8),7)
    image, background = subtract_background_rolling_ball(image, 400, light_background=False,
                                         use_paraboloid=False, do_presmooth=True)
    im_vector = image.reshape((-1,)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(im_vector,k,None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    center = center.astype(np.uint8)
    label_im = label.reshape((image.shape))
    seg_im = np.zeros_like(label_im)
    for i in np.unique(label_im):
        seg_im = np.where(label_im == i, center[i],seg_im)
    
    _, seg_im = cv2.threshold(seg_im.astype(np.uint16), 10, 255, cv2.THRESH_BINARY)
    
    _, seg_im = fill_holes(seg_im.astype(np.uint8),hole_size)
    seg_im = remove_small_objects(seg_im,ditzle_size)
    
    return seg_im

snr_orig=[]
snr_sliding=[]
snr_median = []
snr_rolling = []
snr_preproc = []

for i in range(2,6):
    this_slice = reslice_volume[i,:,:].astype(np.uint8)
    snr_orig.append(vm.signal_to_noise(volume[i,:,:]))
    snr_sliding.append(vm.signal_to_noise(this_slice))
    
    med = cv2.medianBlur(this_slice,7)
    snr_median.append(vm.signal_to_noise(med))
    
    rolling, _ = subtract_background_rolling_ball(this_slice, 400, light_background=False,
                                         use_paraboloid=False, do_presmooth=True)
    snr_rolling.append(vm.signal_to_noise(rolling))
    
    preproc, _ = subtract_background_rolling_ball(med, 400, light_background=False,
                                         use_paraboloid=False, do_presmooth=True)
    snr_preproc.append(vm.signal_to_noise(preproc))

labels = ['orig', 'sliding', 'median', 'rolling', 'preprocess']
means = [np.mean(snr_orig), np.mean(snr_sliding), np.mean(snr_median), np.mean(snr_rolling), np.mean(snr_preproc)]
plt.figure()
plt.bar(labels,means)    


test_out = segment_vessels(this_slice
                           )

plt.figure(); plt.imshow(test_slice2)
plt.figure(); plt.imshow(bigger_thresh2)

output = segment_vessels(test_slice2)
    