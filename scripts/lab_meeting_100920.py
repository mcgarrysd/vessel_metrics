#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:16:38 2020

script used to generate figures for 10/09/2020 presentation

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

img_path = '/home/sean/Documents/Calgary_postdoc/presentation_images/'

plt.close('all')
with CziFile('/home/sean/Documents/Calgary_postdoc/Data/zebrafish_with_dye_091320/56hpf20xDextran4.czi') as czi:
    image_arrays = czi.asarray()
    
image = np.squeeze(image_arrays)

dye_image = image[0,:,:,:]
transgene_image = image[1,:,:,:]

dye_image = dye_image.astype(np.uint8)
transgene_image = transgene_image.astype(np.uint8)

dye_max = intensity_projection(dye_image,4, 'max')

dye_slice = dye_max[17,:,:].astype(np.uint8)
cv2.imwrite(img_path + 'dye_slice.png',dye_slice)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
dye_slice_clahe = clahe.apply(dye_slice)
cv2.imwrite(img_path + 'dye_slice_clahe.png',dye_slice_clahe)

dye_slice_denoise = cv2.medianBlur(dye_slice, 3)
cv2.imwrite(img_path + 'dye_slice_denoise.png',dye_slice_denoise)

_, dye_slice_thresh = cv2.threshold(dye_slice, 25, 255, cv2.THRESH_BINARY)
cv2.imwrite(img_path + 'dye_slice_thresh.png',dye_slice_thresh)

connectivity = 8
num_labels, stat_labels, stats, centroids = cv2.connectedComponentsWithStats(dye_slice_thresh,connectivity, cv2.CV_32S)
    
img_mask = np.zeros(dye_slice_thresh.shape)
    
for i in range(num_labels-1):
    if stats[i+1,4] > 300:
        img_mask[stat_labels == i+1] = 255

img_mask = img_mask.astype(np.uint8)
cv2.imwrite(img_path + 'dye_slice_conncomp.png',img_mask)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
dye_slice_close = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel, iterations = 1)
cv2.imwrite(img_path + 'dye_slice_close.png',dye_slice_close)
