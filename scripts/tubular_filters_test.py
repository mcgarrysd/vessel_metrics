#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:45:04 2021

Tubular filters

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skan import draw
from skan import skeleton_to_csgraph
from scipy.spatial import distance
import vessel_metrics as vm
from skimage import draw
from skimage.filters import meijering, hessian, frangi, sato
from cv2_rolling_ball import subtract_background_rolling_ball

input_directory = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/'
file_name = '35M-59H inx 48hpf Apr 14 2019 E2.czi'

image = vm.preprocess_czi(input_directory,file_name)

view_slices = False
if view_slices:
    plt.figure()
    plt.subplot(5,1,1)
    plt.imshow(image[5,:,:])
    plt.subplot(5,1,2)
    plt.imshow(image[10,:,:])
    plt.subplot(5,1,3)
    plt.imshow(image[15,:,:])
    plt.subplot(5,1,4)
    plt.imshow(image[20,:,:])
    plt.subplot(5,1,5)
    plt.imshow(image[25,:,:])
    
mip = vm.czi_projection(image,0)
plt.figure();
plt.imshow(mip)

reslice_image = vm.reslice_image(image,4)
view_reslice = False
if view_reslice:
    plt.figure()
    plt.subplot(5,1,1)
    plt.imshow(reslice_image[2,:,:])
    plt.subplot(5,1,2)
    plt.imshow(reslice_image[3,:,:])
    plt.subplot(5,1,3)
    plt.imshow(reslice_image[4,:,:])
    plt.subplot(5,1,4)
    plt.imshow(reslice_image[5,:,:])
    plt.subplot(5,1,5)
    plt.imshow(reslice_image[6,:,:])

test_slice = reslice_image[4,:,:]

vessel_filters = False
if vessel_filters:
    median_im = cv2.medianBlur(test_slice.astype(np.uint8),7)    
    Z = median_im.reshape((-1,)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = 16
    ret, label, center = cv2.kmeans(Z,k,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = center.astype(np.uint8)
    label_im = label.reshape((median_im.shape))
    seg_im = np.zeros_like(label_im)
    for i in np.unique(label_im):
        seg_im = np.where(label_im == i, center[i],seg_im)
    
    ret, thresh_binary = cv2.threshold(seg_im.astype(np.uint16), 1, 255, cv2.THRESH_BINARY)
    ret, thresh_bin2 = cv2.threshold(median_im.astype(np.uint16), 1,255, cv2.THRESH_BINARY)
    
    sato_im = sato(seg_im, sigmas = range(1,10,1), mode = 'reflect', black_ridges = False)
    frangi_im = frangi(seg_im, sigmas = range(1,10,1))

##################################################################
test_seg = False
if test_seg:
    test_im = cv2.medianBlur(test_slice.astype(np.uint8),7)
    test_im, background = subtract_background_rolling_ball(test_im, 200, light_background=False,
                                         use_paraboloid=False, do_presmooth=True)
    im_vector = test_im.reshape((-1,)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = 12
    ret, label, center = cv2.kmeans(im_vector,k,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = center.astype(np.uint8)
    label_im = label.reshape((test_im.shape))
    seg_im = np.zeros_like(label_im)
    for i in np.unique(label_im):
        seg_im = np.where(label_im == i, center[i],seg_im)
    
    ret, seg_im = cv2.threshold(seg_im.astype(np.uint16), 1, 255, cv2.THRESH_BINARY)
    
    skel, seg_im = vm.fill_holes(seg_im.astype(np.uint8),200)
    seg_im = vm.remove_small_objects(seg_im,750)
    
##############################################################################

# seg_im = vm.segment_vessels(test_slice,k = 12, hole_size = 200, ditzle_size = 750)

slice1 = reslice_image[3,:,:]
slice2 = reslice_image[4,:,:]
slice3 = reslice_image[5,:,:]
slice4 = reslice_image[6,:,:]

display_seg = False
if display_seg:
    seg1 = vm.segment_vessels(slice1 ,k = 12, hole_size = 200, ditzle_size = 750)
    seg2 = vm.segment_vessels(slice2 ,k = 12, hole_size = 200, ditzle_size = 750)
    seg3 = vm.segment_vessels(slice3 ,k = 12, hole_size = 200, ditzle_size = 750)
    seg4 = vm.segment_vessels(slice4 ,k = 12, hole_size = 200, ditzle_size = 750)
    
    plt.figure(); plt.imshow(slice1)
    plt.figure(); plt.imshow(seg1)
    
    plt.figure(); plt.imshow(slice2)
    plt.figure(); plt.imshow(seg2)
    
    plt.figure(); plt.imshow(slice3)
    plt.figure(); plt.imshow(seg3)
    
    plt.figure(); plt.imshow(slice4)
    plt.figure(); plt.imshow(seg4)

reslice_sliding = vm.sliding_window(image,4)



plt.figure(); plt.imshow(reslice_sliding[14,:,:])
plt.figure(); plt.imshow(reslice_sliding[15,:,:])
plt.figure(); plt.imshow(reslice_sliding[16,:,:])

plt.figure();
