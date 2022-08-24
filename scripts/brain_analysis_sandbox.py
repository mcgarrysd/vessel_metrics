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
#data_files = os.listdir(data_path)

show_volumes = False
if show_volumes:
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
    
save_projections = False
if save_projections:
    out_path = '/home/sean/Documents/vessel_metrics/data/suchit_wt_projections/'
    for file, volume in zip(data_files, reslice_list):
        im = volume[0]
        fish = file.split('_')[0]
        im_name = fish + '.png'
        cv2.imwrite(out_path +im_name, im)
    
data_path = '/home/sean/Documents/vessel_metrics/data/suchit_wt_projections/'
test_ims = ['emb2.png', 'emb8.png']
for im_name in test_ims:
    im = cv2.imread(data_path+im_name,0)
    label = brain_seg(im)
    plt.figure()
    plt.imshow(label)

plt.imshow(volume[0])
################
def brain_seg(im, hole_size = 50, ditzle_size = 500):
    im = vm.contrast_stretch(im)
    im = vm.preprocess_seg(im)
    _, seg = cv2.threshold(im.astype(np.uint16), 100, 255, cv2.THRESH_BINARY)
    
    sato_im = sato(im, sigmas = range(1,10,2), mode = 'reflect', black_ridges = False)
    sato_norm = np.round(sato_im/np.max(sato_im)*255).astype(np.uint8)
    
    sato_label = np.zeros_like(sato_norm)
    sato_label[sato_norm>40] =1
    
    overlap = sato_label + seg
    
    label = np.zeros_like(seg)
    label[overlap>1] =1
    
    kernel = np.ones((6,6),np.uint8)
    opening = cv2.morphologyEx(label.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    _, label = vm.fill_holes(label.astype(np.uint8),hole_size)
    label = vm.remove_small_objects(label,ditzle_size)
    
    return label


def brain_seg2(im, hole_size = 50, ditzle_size = 500, sato_thresh = 30):
    im = vm.contrast_stretch(im)
    im = vm.preprocess_seg(im)
    
    sato_im = sato(im, sigmas = range(1,10,2), mode = 'reflect', black_ridges = False)
    sato_norm = np.round(sato_im/np.max(sato_im)*255).astype(np.uint8)
    
    sato_label = np.zeros_like(sato_norm)
    sato_label[sato_norm>sato_thresh] =1
    
    
    kernel = np.ones((6,6),np.uint8)
    label = cv2.morphologyEx(sato_label.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    _, label = vm.fill_holes(label.astype(np.uint8),hole_size)
    label = vm.remove_small_objects(label,ditzle_size)
    
    return label

# Jaccard on image 8
seg = brain_seg2(im, sato_thresh = 60)
label = cv2.imread('/home/sean/Documents/suchit_wt_projections/emb8/label.png',0)
label[label>0] =1

jacc = vm.jaccard(label,seg)
overlap = seg+2*label

##################### Do something useful with an image

skel = skeletonize(label2)

# need other channel from mutant data
mt_skel = skeletonize(mt_label)
mt_bp = branchpoint_density(mt_skel,mt_label)

plt.figure()
plt.boxplot([bp,mt_bp], labels = ['wild type', 'mutant'])


def overlay_segmentation(im,label, alpha = 0.5, contrast_stretch = False):
    if contrast_stretch:
        im = vm.contrast_stretch(im)
        im = vm.preprocess_seg(im)
    masked = np.ma.masked_where(label == 0, label)
    plt.figure()
    plt.imshow(im, 'gray', interpolation = 'none')
    plt.imshow(masked, 'jet', interpolation = 'none', alpha = alpha)
    plt.show()
    
########################### mutant projections

mt_path = '/home/sean/Documents/suchit_mt/'
data_files = os.listdir(mt_path)

save_projections = True
if save_projections:
    reslice_list = []
    for file in data_files:
        volume = vm.preprocess_czi(mt_path,file, channel = 1)
        slice_range = len(volume)
        slice_thickness = np.round(slice_range/2).astype(np.uint8)
        reslice = vm.reslice_image(volume,slice_thickness)
        reslice_list.append(reslice)
    
    
    out_path = '/home/sean/Documents/vessel_metrics/data/suchit_mt_projections/'
    for file, volume in zip(data_files, reslice_list):
        im = volume[0]
        fish = file.split('_')[0]
        im_name = fish + '.png'
        cv2.imwrite(out_path +im_name, im)