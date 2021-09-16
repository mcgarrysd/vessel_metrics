#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 07:26:41 2021

brain vessel diameter

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
from scipy.spatial import distance

wt_path = '/home/sean/Documents/vessel_metrics/data/suchit_wt_projections/'
wt_names = ['emb9']
wt_ims = []
wt_seg = []
for im_name in wt_names:
    im = cv2.imread(wt_path+im_name+'.png',0)
    wt_ims.append(im)
    wt_seg.append(vm.brain_seg(im))
    
im = wt_ims[0]
seg = wt_seg[0]

im_crop = vm.crop_brain_im(im)
seg_crop= vm.crop_brain_im(seg)

skel = skeletonize(seg_crop)
edges, bp = vm.find_branchpoints(skel)

_, edge_labels = cv2.connectedComponents(edges)

this_seg = 9
segment = np.zeros_like(edge_labels)
segment[edge_labels==this_seg] = 1

segment_median = segment_midpoint(segment)
distance_im = segment_distance(segment)



plt.imshow(edge_labels)

#######################################################################
def segment_distance(segment):
    segment_endpoints = vm.find_endpoints(segment)
    endpoint_index = np.where(segment_endpoints)
    first_endpoint = endpoint_index[0][0], endpoint_index[1][0]
    segment_indexes = np.argwhere(segment==1)
        
    distances = []
    for i in range(len(segment_indexes)):
        this_pt = segment_indexes[i][0], segment_indexes[i][1]
        distances.append(distance.chebyshev(first_endpoint, this_pt))
    distance_im =np.zeros_like(segment)
    for i in range(len(distances)):
        distance_im[segment_indexes[i][0], segment_indexes[i][1]]=distances[i]
    return distance_im

def segment_midpoint(segment):
    segment_endpoints = vm.find_endpoints(segment)
    endpoint_index = np.where(segment_endpoints)
    first_endpoint = endpoint_index[0][0], endpoint_index[1][0]
    segment_indexes = np.argwhere(segment==1)
        
    distances = []
    for i in range(len(segment_indexes)):
        this_pt = segment_indexes[i][0], segment_indexes[i][1]
        distances.append(distance.chebyshev(first_endpoint, this_pt))
    sort_indexes = np.argsort(distances)
    sorted_distances = sorted(distances)
    median_distance= np.where(sorted_distances == np.median(sorted_distances))[0][0]
    segment_median = segment_indexes[np.where(distances == median_distance)]
    
    return segment_median

def tangent_slope(segment, point)
