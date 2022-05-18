#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:45:15 2021

grant figure 09/02/21

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
from scipy import stats

fiugre_path = '/home/sean/Documents/Calgary_postdoc/figures/'

wt_path = '/home/sean/Documents/suchit_wt_projections/'
wt_names = ['emb3', 'emb6', 'emb8', 'emb9']
wt_ims = []
wt_labels = []
wt_seg = []
for im_name in wt_names:
    im = cv2.imread(wt_path+im_name+'/img.png',0)
    wt_ims.append(im)
    wt_labels.append(cv2.imread(wt_path+im_name+'/label.png',0))
    wt_seg.append(vm.brain_seg(im))
    
mt_path = '/home/sean/Documents/vessel_metrics/data/suchit_mt_projections/'
mt_names = ['emb3', 'emb4', 'emb5', 'emb13','emb15']
mt_ims = []
mt_labels = []
mt_seg = []
for im_name in mt_names:
    im = cv2.imread(mt_path+im_name+'/img.png',0)
    mt_ims.append(im)
    mt_labels.append(cv2.imread(mt_path+im_name+'/label.png',0))
    mt_seg.append(vm.brain_seg(im))

plt.figure()
plt.imshow(wt_ims[1], 'gray')
plt.imshow(mt_ims[0],'gray')

vm.overlay_segmentation(wt_ims[1], wt_seg[1])
vm.overlay_segmentation(mt_ims[0], mt_seg[0])


def vessel_density(im,label, num_tiles_x, num_tiles_y):
    density = np.zeros_like(im).astype(np.float16)
    density_array = []
    label[label>0] = 1
    step_x = np.round(im.shape[0]/num_tiles_x).astype(np.int16)
    step_y = np.round(im.shape[1]/num_tiles_y).astype(np.int16)
    for x in range(0,im.shape[0], step_x):
        for y in range(0,im.shape[1], step_y):
            tile = label[x:x+step_x-1,y:y+step_y-1]
            numel = tile.shape[0]*tile.shape[1]
            tile_density = np.sum(tile)/numel
            tile_val = np.round(tile_density*1000)
            density[x:x+step_x-1,y:y+step_y-1] = tile_val
            density_array.append(tile_val)
    density = density.astype(np.uint16)
    return density, density_array

nz_wt = np.nonzero(wt_density)
nz_mt = np.nonzero(mt_density)

plt.figure()
plt.boxplot([nz_wt[0],nz_mt[0]], labels = ['wild type', 'mutant'])

wt_16 = vessel_density(wt_ims[1], wt_seg[1], 16,16)
mt_16 = vessel_density(mt_ims[0], mt_seg[0], 16,16)

nz_wt_16 = np.nonzero(wt_16[1])
nz_mt_16 = np.nonzero(mt_16[1])

plt.figure()
plt.boxplot([nz_wt_16[0], nz_mt_16[0]], labels = ['wild type', 'mutant'])

vm.overlay_segmentation(im,density, alpha = 0.2)

def crop_brain_im(im,label = None):
    new_im = im[300:750,50:500]
    if label is None:
        return new_im
    else:
        new_label = label[300:750,50:500]
        return new_im, label

    
    