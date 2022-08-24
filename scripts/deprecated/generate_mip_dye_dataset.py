#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:28:04 2020

Takes czi files, reslices with maximum intensity projection
and saves image files by slice to be labelled at a later time

@author: sean
"""

from czifile import CziFile
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

data_path = '/home/sean/Documents/Calgary_postdoc/Data/zebrafish_with_dye_091320/'
file_list = ['fish1', 'fish2', 'fish3', 'fish4', 'fish5', 'fish6'] 

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))

for i in file_list:
    file_name = i + '.czi'
    with CziFile(data_path + file_name) as czi:
        image_arrays = czi.asarray()
    
    image_arrays = np.squeeze(image_arrays)
    transgene_image = image_arrays[0,:,:,:]
    dye_image = image_arrays[1,:,:,:]
    
    transgene_max = intensity_projection(transgene_image,4, 'max')
    dye_max = intensity_projection(dye_image,4, 'max')
    
    for j in range(transgene_max.shape[0]):
        transgene_slice = np.squeeze(transgene_max[j,:,:])
        dye_slice = np.squeeze(dye_max[j,:,:])
        
        transgene_slice = clahe.apply(transgene_slice.astype(np.uint8))
        dye_slice = clahe.apply(dye_slice.astype(np.uint8))
        
        image_name = i + '_slice_' + str(j) + '.png'
        
        cv2.imwrite(data_path + 'dye_images/'+ image_name, dye_slice)
        cv2.imwrite(data_path + 'transgene_images/'+ image_name, transgene_slice)
    