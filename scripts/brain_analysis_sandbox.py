#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 09:30:27 2021

brain analysis sandbox
sandbox for figure requested by schilds for grand submission fall 2021

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


plt.close('all')

data_path = '/home/sean/Documents/suchit_feb_21/'
data_files = os.listdir(data_path)

file = data_files[0]
for file in data_files:
    volume = vm.preprocess_czi(data_path,file)
    for i in range(0,len(volume),5):
        plt.figure()
        plt.imshow(volume[i])

reslice_list = []
for file in data_files:
    volume = vm.preprocess_czi(data_path,file)
    slice_range = len(volume)
    slice_thickness = np.round(slice_range/2).astype(np.uint8)
    reslice = vm.reslice_image(volume,slice_thickness)
    reslice_list.append(reslice)

for i in reslice_list:
    plt.figure(); plt.imshow(i[0])
    
out_path = '/home/sean/Documents/vessel_metrics/data/suchit_wt_projections/'
for file, volume in zip(data_files, reslice_list):
    im = volume[0]
    fish = file.split('_')[0]
    im_name = fish + '.png'
    cv2.imwrite(out_path +im_name, im)
    
test_ims = ['emb2.png', 'emb8.png']
for im_name in test_ims:
    im = cv2.imread(out_path+im_name,0)
    label = brain_seg(im)
    plt.figure()
    plt.imshow(label)
    



################
def brain_seg(im, hole_size = 50, ditzle_size = 500):
    im = vm.contrast_stretch(im)
    im = vm.preprocess_seg(im)
    _, seg = cv2.threshold(im.astype(np.uint16), 100, 255, cv2.THRESH_BINARY)
    
    sato_im = sato(im, sigmas = range(1,10,2), mode = 'reflect', black_ridges = False)
    sato_norm = np.round(sato_im/np.max(sato_im)*255).astype(np.uint8)
    
    sato_label = np.zeros_like(sato_norm)
    sato_label[sato_norm>15] =1
    
    overlap = sato_label + seg
    
    label = np.zeros_like(seg)
    label[overlap>1] =1
    
    _, label = vm.fill_holes(label.astype(np.uint8),hole_size)
    label = vm.remove_small_objects(label,ditzle_size)
    
    return label
    