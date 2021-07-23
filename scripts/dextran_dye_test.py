#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:27:47 2020

Preliminary work with zebrafish embryo post dye injection

@author: sean
"""

from czifile import CziFile
import matplotlib.pyplot as plt
import numpy as np
import cv2

def trunk_segmentation(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 11)
    _, th1 = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,1)
    
    img[th1==0]=0;
    connectivity = 8
    num_labels, stat_labels, stats, centroids = cv2.connectedComponentsWithStats(img,connectivity, cv2.CV_32S)
    
    img_mask = np.zeros(img.shape)
    
    for i in range(num_labels-1):
        if stats[i+1,4] > 500:
            img_mask[stat_labels == i+1] = 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel, iterations = 1)
    
    return img_mask

def make_slices(img, step_size):
    steps = list(range(img.shape[0]))
    slice_count = 0
    num_steps = np.int(np.ceil(img.shape[0]/step_size))
    output = np.zeros([num_steps,img.shape[1],img.shape[2]])
    for i in steps[::step_size]:
        temp_img = img[i:i+step_size,:,:]
        new_slice = np.mean(temp_img, axis = 0)
        output[slice_count,:,:] = new_slice
        slice_count += 1
    return output

def intensity_projection(img,step_size,method):
    steps = list(range(img.shape[0]))
    slice_count = 0
    num_steps = np.int(np.ceil(img.shape[0]/step_size))
    output = np.zeros([num_steps,img.shape[1],img.shape[2]])
    for i in steps[::step_size]:
        temp_img = img[i:i+step_size,:,:]
        if method == 'mean':
            new_slice = np.mean(temp_img, axis = 0)
        if method == 'max':
            new_slice = np.max(temp_img, axis = 0)
        if method == 'median':
            new_slice = np.median(temp_img, axis = 0)
        if method == 'std':
            new_slice = np.std(temp_img, axis = 0)
        if method == 'sum':
            new_slice = np.sum(temp_img, axis = 0)
            new_slice[new_slice>255]=255
        output[slice_count,:,:] = new_slice
        slice_count += 1
    return output

plt.close('all')
with CziFile('/home/sean/Documents/Calgary_postdoc/Data/zebrafish_with_dye_091320/56hpf20xDextran4.czi') as czi:
    image_arrays = czi.asarray()
    
image = np.squeeze(image_arrays)

dye_image = image[0,:,:,:]
transgene_image = image[1,:,:,:]

mip_transgene = np.max(transgene_image, axis = 0)
mip_dye = np.max(dye_image, axis = 0)

transgene_seg = trunk_segmentation(mip_transgene)
dye_seg = trunk_segmentation(mip_dye)

num_slices = list(range(dye_image.shape[0]))

# SEGMENTATION APPLIED TO INDIVIDUAL SLICES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

segmentation_flag = 0
if segmentation_flag == 1:
    plt.figure(1); plt.figure(2)
    plot_count = 0
    for x in num_slices[0::15]:
        plot_count += 1
        plt.figure(1)
        plt.subplot(7,2,plot_count)
        dye_slice = dye_image[x,:,:]
        plt.imshow(dye_slice)
        
        plt.figure(2)
        plt.subplot(7,2,plot_count)
        transgene_slice = transgene_image[x,:,:]
        plt.imshow(transgene_slice)
        
        plot_count += 1
        dye_slice_seg = trunk_segmentation(dye_slice)
        transgene_slice_seg = trunk_segmentation(transgene_slice)
        
        plt.figure(1)
        plt.subplot(7,2,plot_count)
        plt.imshow(dye_slice_seg)
        
        plt.figure(2)
        plt.subplot(7,2,plot_count)
        plt.imshow(transgene_slice_seg)



# SEGMENTATION APPLIED TO COMBINED SLICES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
dye_resliced = make_slices(dye_image,4)
transgene_resliced = make_slices(transgene_image,4)

num_combined_slices = list(range(dye_resliced.shape[0]))

reslice_flag = 0
if reslice_flag == 1:
    plt.figure(3); plt.figure(4)
    plot_count = 0
    for x in num_combined_slices[0::4]:
        plot_count += 1
        plt.figure(3)
        plt.subplot(6,2,plot_count)
        dye_slice = dye_image[x,:,:]
        plt.imshow(dye_slice)
        
        plt.figure(4)
        plt.subplot(6,2,plot_count)
        transgene_slice = transgene_image[x,:,:]
        plt.imshow(transgene_slice)
        
        plot_count += 1
        dye_slice_seg = trunk_segmentation(dye_slice)
        transgene_slice_seg = trunk_segmentation(transgene_slice)
        
        plt.figure(3)
        plt.subplot(6,2,plot_count)
        plt.imshow(dye_slice_seg)
        
        plt.figure(4)
        plt.subplot(6,2,plot_count)
        plt.imshow(transgene_slice_seg)

# Display different intensity projections
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dye_mean = intensity_projection(dye_image, 4, 'mean')
dye_median = intensity_projection(dye_image,4, 'median')
dye_max = intensity_projection(dye_image,4, 'max')
dye_std = intensity_projection(dye_image,4, 'std')
dye_sum = intensity_projection(dye_image,4, 'sum')

trans_mean = intensity_projection(transgene_image, 4, 'mean')
trans_median = intensity_projection(transgene_image,4, 'median')
trans_max = intensity_projection(transgene_image,4, 'max')
trans_std = intensity_projection(transgene_image,4, 'std')
trans_sum = intensity_projection(transgene_image,4, 'sum')

num_combined_slices = list(range(dye_mean.shape[0]))
display_slice = 17

projection_flag = 0
if projection_flag == 1:
    plt.figure(5);
    plot_count = 0
    
    plt.subplot(5,2,1)
    plt.imshow(dye_mean[display_slice,:,:])
    plt.subplot(5,2,2)
    plt.imshow(trans_mean[display_slice,:,:])
    plt.subplot(5,2,3)
    plt.imshow(dye_median[display_slice,:,:])
    plt.subplot(5,2,4)
    plt.imshow(trans_median[display_slice,:,:])
    plt.subplot(5,2,5)
    plt.imshow(dye_max[display_slice,:,:])
    plt.subplot(5,2,6)
    plt.imshow(trans_max[display_slice,:,:])
    plt.subplot(5,2,7)
    plt.imshow(dye_std[display_slice,:,:])
    plt.subplot(5,2,8)
    plt.imshow(trans_std[display_slice,:,:])
    plt.subplot(5,2,9)
    plt.imshow(dye_sum[display_slice,:,:])
    plt.subplot(5,2,10)
    plt.imshow(trans_sum[display_slice,:,:])

# Combine the dye and transgene images
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

combination = dye_max + trans_max
combination[combination>255] = 255

new_combination = np.zeros([3,combination.shape[1], combination.shape[2]])
new_combination[0,:,:] = dye_max[display_slice,:,:]
new_combination[1,:,:] = trans_max[display_slice,:,:]
new_combination = np.moveaxis(new_combination,0,-1)

new_combination_mean = np.zeros([3,combination.shape[1], combination.shape[2]])
new_combination_mean[0,:,:] = dye_mean[display_slice,:,:]
new_combination_mean[1,:,:] = trans_mean[display_slice,:,:]
new_combination_mean = np.moveaxis(new_combination_mean,0,-1)

combination_flag = 1
if combination_flag == 1:
    plt.figure();
    plt.imshow(combination[display_slice,:,:])
    
    plt.figure();
    plt.imshow(new_combination)
    


# Display base images and maximum intensity projection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
display_flag = 0

if display_flag:
    plt.figure()
    plt.subplot(1,2,1); plt.imshow(mip_transgene)
    plt.title('transgene MIP')
    plt.subplot(1,2,2); plt.imshow(transgene_seg)
    plt.title('transgene segmentation')
    
    plt.figure()
    plt.subplot(1,2,1); plt.imshow(mip_dye)
    plt.title('dye MIP')
    plt.subplot(1,2,2); plt.imshow(dye_seg)
    plt.title('dye segmentation')



