#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:32:23 2021

overlapping_vessels_sandbox

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skan import draw
from skan import skeleton_to_csgraph
import vessel_metrics as vm

input_directory = '/home/sean/Documents/Calgary_postdoc/Data/raw_czi/'
file_name = 'flk gata inx 30hpf Feb 23 2019 E1 good qual.czi'
label_dir = '/home/sean/Documents/Calgary_postdoc/Data/czi_projections/Channel0/contrast_enhanced/E1_good_qual/'
label_name = 'label.png'

label = cv2.imread(label_dir + label_name,0)
label[label>0] = 1
skel, label = vm.fill_holes(label, 50)

branch_points, edges = vm.find_branchpoints(skel)
num_labels, edge_labels = cv2.connectedComponents(edges, connectivity = 8)
edge_labels, edges = vm.remove_small_segments(edge_labels, 5)
test_segment = 139

chunk = vm.segment_chunk(test_segment, edge_labels, volume)
x_chunk = vm.czi_projection(chunk,1)
y_chunk = vm.czi_projection(chunk,2)
z_chunk = vm.czi_projection(chunk,0)

display_chunk = False
if display_chunk:
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x_chunk)
    plt.subplot(1,3,2)
    plt.imshow(y_chunk)
    plt.subplot(1,3,3)
    plt.imshow(z_chunk)

slices_y = True
if slices_y:
    plt.figure()
    for i in range(1,11):
        plt.subplot(5,2,i)
        plt.imshow(chunk[:,:,i*10])
        
im_slice = chunk[:,:,30]
plt.figure(); plt.imshow(im_slice)
im_slice = im_slice.astype(np.uint8)

plot_slice_thresh = False
if plot_slice_thresh:
    binary_slice = np.zeros_like(im_slice)
    binary_slice[im_slice>0] = 1
    plt.figure();
    plt.subplot(2,2,1); plt.imshow(binary_slice)
    ret1,th_global = cv2.threshold(im_slice,5,255,cv2.THRESH_BINARY)
    plt.subplot(2,2,2); plt.imshow(th_global)
    ret2,th_otsu = cv2.threshold(im_slice,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.subplot(2,2,3); plt.imshow(th_otsu)
    blur = cv2.GaussianBlur(im_slice,(5,5),0)
    ret3,th_blue = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.subplot(2,2,4); plt.imshow(th_otsu)
    
    th_adapt = cv2.adaptiveThreshold(im_slice,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    plt.figure(); plt.imshow(th_adapt)

projection = np.max(chunk[:,:,10:30], axis = 2)
projection = projection.astype(np.uint8)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast_projection = clahe.apply(projection)

plot_proj_thresh = True
if plot_proj_thresh:
    binary_slice = np.zeros_like(im_slice)
    binary_slice[projection>0] = 1
    plt.figure();
    plt.subplot(2,2,1); plt.imshow(binary_slice)
    ret1,th_global = cv2.threshold(projection,10,255,cv2.THRESH_BINARY)
    plt.subplot(2,2,2); plt.imshow(th_global)
    ret2,th_otsu = cv2.threshold(projection,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.subplot(2,2,3); plt.imshow(th_otsu)
    blur = cv2.GaussianBlur(projection,(5,5),0)
    ret3,th_blue = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.subplot(2,2,4); plt.imshow(th_otsu)
    
    th_adapt = cv2.adaptiveThreshold(projection,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    plt.figure(); plt.imshow(th_adapt)



hough_circle = True
if hough_circle:
    circles = cv2.HoughCircles(contrast_projection,cv2.HOUGH_GRADIENT,1,20,
                            param1=100,param2=10,minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))
    circle_disp = projection.copy()
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(circle_disp,(i[0],i[1]),i[2],255,2)
        # draw the center of the circle
        cv2.circle(circle_disp,(i[0],i[1]),2,255,3)
    plt.imshow(circle_disp)

slices_x = True
if slices_x:
    plt.figure()
    for i in range(1,11):
        plt.subplot(5,2,i)
        plt.imshow(chunk[:,i*17,:])


volume = vm.preprocess_czi(input_directory, file_name)

x_proj = vm.czi_projection(volume,1)
y_proj = vm.czi_projection(volume,2)
z_proj = vm.czi_projection(volume,0)

display_projections = False
if display_projections:
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x_proj)
    plt.subplot(1,3,2)
    plt.imshow(y_proj)
    plt.subplot(1,3,3)
    plt.imshow(z_proj)
    

