#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:33:42 2022

Murine data test

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


data_path = '/media/sean/SP PHD U3/from_home/murine_data/milene/'

test_im = 'Z.tif'

im = cv2.imread(data_path+test_im,0)
vm.show_im(im)

preproc = vm.preprocess_seg(im)
vm.show_im(preproc)

seg = vm.brain_seg(im, sigmas = range(1,20,2), filter = 'meijering', thresh = 20, preprocess = False)
vm.overlay_segmentation(im, seg)


seg_nt_big = vm.seg_no_thresh(im, sigmas = range(10,20,5), filter = 'meijering')

seg_nt_sm = vm.seg_no_thresh(im, sigmas = range(1,5,1), filter = 'meijering')
seg_nt_test = vm.normalize_contrast(seg_nt_big.astype(np.uint16)+seg_nt_sm.astype(np.uint16))

vm.show_im(seg_nt_big)
vm.show_im(seg_nt_sm)
vm.show_im(seg_nt_test)

thresh = 40
seg = np.zeros_like(seg_nt_big)
seg[seg_nt_test>thresh] =1
vm.overlay_segmentation(im, seg)

skel, edges, bp = vm.skeletonize_vm(seg)
vm.overlay_segmentation(im, edges)
_, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
vm.overlay_segmentation(im, edge_labels)

seg_number = 10
diam, mean_diam, viz = vm.visualize_vessel_diameter(edge_labels, seg_number, seg, im)
vm.overlay_segmentation(im, viz)

test2 = multi_scale_seg(im, sigma1 = range(1,4,1), sigma2 = range(10,20,5), hole_size = 200, ditzle_size = 0)

data_path = '/media/sean/SP PHD U3/from_home/murine_data/milene/'
out_path = '/media/sean/SP PHD U3/from_home/murine_data/milene_seg/'

files = os.listdir(data_path)
for file in files:
    im = cv2.imread(data_path+file,0)
    seg = multi_scale_seg(im, sigma1 = range(1,5,1), ditzle_size = 0, hole_size = 200)
    vm.overlay_segmentation(im, seg)
    out_name = file.split('.')[0]+'.png'
    plt.savefig(out_path+out_name,bbox_inches='tight')
    plt.close('all')
    

data_path = '/media/sean/SP PHD U3/from_home/murine_data/adam/'
out_path = '/media/sean/SP PHD U3/from_home/murine_data/adam_seg/'

files = os.listdir(data_path)
for file in files:
    im = cv2.imread(data_path+file,0)
    seg = multi_scale_seg(im, sigma1 = range(1,5,1), ditzle_size = 0, hole_size = 200)
    vm.overlay_segmentation(im, seg)
    out_name = file.split('.')[0]+'.png'
    plt.savefig(out_path+out_name,bbox_inches='tight')
    plt.close('all')

def multi_scale_seg(im, filter = 'meijering', sigma1 = range(1,8,1), sigma2 = range(10,20,5), hole_size = 50, ditzle_size = 500, thresh = 40, preprocess = True):
    if preprocess == True:
        im = preprocess_seg(im)
    
    if filter == 'meijering':
        enh_sig1 = meijering(im, sigmas = sigma1, mode = 'reflect', black_ridges = False)
        enh_sig2 = meijering(im, sigmas = sigma2, mode = 'reflect', black_ridges = False)
    elif filter == 'sato':
        enhanced_im = sato(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'frangi':
        enh_sig1 = frangi(im, sigmas = sigma1, mode = 'reflect', black_ridges = False)
        enh_sig2 = frangi(im, sigmas = sigma2, mode = 'reflect', black_ridges = False)
    elif filter == 'jerman':
        enh_sig1 = vm.jerman(im, sigmas = sigma1, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')
        enh_sig2 = vm.jerman(im, sigmas = sigma1, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')
    sig1_norm = vm.normalize_contrast(enh_sig1)
    sig2_norm = vm.normalize_contrast(enh_sig2)
    
    norm = sig1_norm.astype(np.uint16)+sig2_norm.astype(np.uint16)
    enhanced_label = np.zeros_like(norm)
    enhanced_label[norm>thresh] =1
    
    
    kernel = np.ones((6,6),np.uint8)
    label = cv2.morphologyEx(enhanced_label.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    _, label = fill_holes(label.astype(np.uint8),hole_size)
    label = remove_small_objects(label,ditzle_size)
    
    return label