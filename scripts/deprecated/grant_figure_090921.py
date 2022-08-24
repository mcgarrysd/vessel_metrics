#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 10:37:40 2021

grant_figure_090921

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

wt_path = '/home/sean/Documents/vessel_metrics/data/suchit_wt_projections/'
wt_names = ['emb3', 'emb6', 'emb8', 'emb9']
wt_ims = []
wt_seg = []
for im_name in wt_names:
    im = cv2.imread(wt_path+im_name+'.png',0)
    wt_ims.append(im)
    wt_seg.append(vm.brain_seg(im))
    
mt_path = '/home/sean/Documents/vessel_metrics/data/suchit_mt_projections/'
mt_names = ['emb3', 'emb4', 'emb5', 'emb13','emb15']
mt_ims = []
mt_labels = []
mt_seg = []
for im_name in mt_names:
    im = cv2.imread(mt_path+im_name+'.png',0)
    mt_ims.append(im)
    mt_labels.append(cv2.imread(mt_path+im_name+'/label.png',0))
    mt_seg.append(vm.brain_seg(im))
    
    
#####################################################################
# Bright and Dim segmentation
bright_im = wt_ims[3]
bright_seg = wt_seg[3]

dim_im = mt_ims[4]
dim_seg = mt_seg[4]

vm.overlay_segmentation(bright_im, bright_seg, alpha = 0.2)
vm.overlay_segmentation(dim_im, dim_seg, alpha = 0.2)

contrast_adjusted = vm.contrast_stretch(dim_im)
plt.imshow(contrast_adjusted)

####################################################################
# vessel diameter test

crop_wt, crop_seg = vm.crop_brain_im(bright_im, bright_seg)
vm.overlay_segmentation(crop_wt, crop_seg, alpha = 0.2)

skel = skeletonize(crop_seg)
vm.overlay_segmentation(crop_wt, skel, alpha = 0.5)

edges, bp = vm.find_branchpoints(skel)
_, edge_labels = cv2.connectedComponents(edges)
end_points = vm.find_endpoints(edges)

segment_number = 9
test_segment = np.zeros_like(crop_wt)
test_segment[edge_labels == segment_number] = 1


#####################################################################
# Diameter measurement functions

def find_segment_midpoint(segment_label):
    segment_endpoints = vm.find_endpoints(test_segment)
    endpoint_index = np.where(segment_endpoints)
    first_endpoint = endpoint_index[0][0], endpoint_index[1][0]

    distance_segment = np.zeros_like(edges)
    distances = []
    for i in range(len(segment_index[0])):
        this_pt = segment_index[0][i], segment_index[1][i]
        distances.append(distance.chebyshev(first_endpoint, this_pt))
        distance_segment[this_pt[0],this_pt[1]] = distance.chebyshev(first_endpoint, this_pt)
        sort_indexes = np.argsort(distances)
        sorted_distances = sorted(distances)


