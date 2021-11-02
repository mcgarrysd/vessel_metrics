#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:52:02 2021

FA dataset

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
from skimage.draw import line
from copy import deepcopy


data_path = '/home/sean/Documents/RECOVERY-FA19/'

im_name = 'Img01_RECOVERY-FA19.tif'
label_name = 'Label01_RECOVERY-FA19.png'


im = cv2.imread(data_path+'Images_RECOVERY-FA19/'+im_name,0)
label = cv2.imread(data_path+'Labels_RECOVERY-FA19/'+label_name,0)
label[label>0] =1

skel = skeletonize(label)
edges, bp = vm.find_branchpoints(skel)

_, edge_labels = cv2.connectedComponents(edges)

unique_edges = np.unique(edge_labels)
unique_edges = unique_edges[1:]

pad_size = 50
edge_label_pad = np.pad(edge_labels,pad_size)
label_pad = np.pad(label, pad_size)

minimum_length = 25
full_viz_pad = np.zeros_like(edge_label_pad)
diameters = []
for i in unique_edges:
    seg_length = len(np.argwhere(edge_label_pad == i))
    if seg_length>minimum_length:
        print(i)
        _, temp_diam, temp_viz = visualize_vessel_diameter(edge_label_pad, i, label_pad)
        diameters.append(temp_diam)
        full_viz_pad = full_viz_pad + temp_viz
 
im_shape = edge_label_pad.shape       
full_viz = full_viz_pad[pad_size:im_shape[0]-pad_size,pad_size:im_shape[1]-pad_size]


overlay = label + full_viz + skel
vm.overlay_segmentation(im, overlay)

edge_label_ovl = label+full_viz+edge_labels
vm.overlay_segmentation(im,edge_label_ovl)


########################################################################
    
edge_label_test = deepcopy(edge_labels)
segment_number = 9698
seg_test = deepcopy(label)

edge_label_test = np.pad(edge_label_test,50)
seg_test = np.pad(seg_test,50)

segment = np.zeros_like(edge_label_test)
segment[edge_label_test==segment_number] = 1

#########################################################

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
median_val = np.median(sorted_distances)
dist_from_median = abs(sorted_distances-median_val)
median_distance= np.where(dist_from_median == np.min(dist_from_median))[0][0]
segment_median = segment_indexes[median_distance]
segment_median = segment_median.flatten()

####################################################
point = segment_median
crop_im = segment[point[0]-5:point[0]+5,point[1]-5:point[1]+5]
crop_inds = np.transpose(np.where(crop_im))
tangent = cv2.fitLine(crop_inds,cv2.DIST_L2,0,0.1,0.1)
vx, vy = tangent[0], tangent[1]
bx = -vy
by = vx

#####################################################
dist = 5
diam = 0
im_size = seg_test.shape[0]
while diam == 0:
    dist +=5
    xlen = bx*dist/2
    ylen = by*dist/2

    x1 = int(np.round(point[0]-xlen))
    x2 = int(np.round(point[0]+xlen))

    y1 = int(np.round(point[1]-ylen))
    y2 = int(np.round(point[1]+ylen))

    rr, cc = line(x1,y1,x2,y2)
    cross_index = []
    for r,c in zip(rr,cc):
        cross_index.append([r,c])
    coords = x1,x2,y1,y2
    if all(i<im_size for i in coords):
        print('coords within image bounds, dist ' + str(dist))
        seg_val = []
        for i in cross_index:
            seg_val.append(seg_test[i[0], i[1]])
        steps = np.where(np.roll(seg_val,1)!=seg_val)[0]
        if steps.size>0:
            if steps[0] == 0:
                steps = steps[1:]
            num_steps = len(steps)
            if num_steps == 2:
                diam = abs(steps[1]-steps[0])
        if dist >100:
            print('dist > 100')
            break
    else:
        print('coords not within image bounds')
        break
length = diam*2.5
################################################################
viz = np.zeros_like(seg_test)
diameter = []
segment_inds = np.argwhere(segment)
for i in range(10,len(segment_inds),10):
    point = segment_inds[i]
    crop_im = segment[point[0]-5:point[0]+5,point[1]-5:point[1]+5]
    crop_inds = np.transpose(np.where(crop_im))
    tangent = cv2.fitLine(crop_inds,cv2.DIST_L2,0,0.1,0.1)
    vx, vy = tangent[0], tangent[1]
    bx = -vy
    by = vx
    
    
    xlen = bx*length/2
    ylen = by*length/2

    x1 = int(np.round(point[0]-xlen))
    x2 = int(np.round(point[0]+xlen))

    y1 = int(np.round(point[1]-ylen))
    y2 = int(np.round(point[1]+ylen))

    rr, cc = line(x1,y1,x2,y2)
    cross_index = []
    for r,c in zip(rr,cc):
        cross_index.append([r,c])
    ###############################################
    
    cross_vals = []
    for c in cross_index:
        cross_vals.append(seg_test[c[0], c[1]])
    ###########################################
    steps = np.where(np.roll(cross_vals,1)!=cross_vals)[0]
    if steps.size>0:
        if steps[0] == 0:
            steps = steps[1:]
        num_steps = len(steps)
        if num_steps == 2:
            diam = abs(steps[1]-steps[0])
        else:
            diam = 0
    else:
        diam = 0
    
    ############################################
    if diam == 0:
        val = 5
    else:
        val = 10
    for c in cross_index:
        viz[c[0], c[1]] = val
    diameter.append(diam)
diameter = [x for x in diameter if x != 0]
mean_diameter = np.mean(diameter)
