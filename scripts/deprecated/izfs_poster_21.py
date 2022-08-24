#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:38:46 2021

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
from skimage.morphology import skeletonize
from scipy.spatial import distance
from bresenham import bresenham
from copy import deepcopy
from skan import draw


data_path = '/home/sean/Documents/Calgary_postdoc/posters/IZFS_2021/'


wt_img = cv2.imread(data_path + 'wt_img.png',0)
mutant_img = cv2.imread(data_path + 'mutant_img.png',0)
wt_label = cv2.imread(data_path + 'wt_label.png',0)
mutant_label = cv2.imread(data_path + 'mutant_label.png',0)

wt_seg = vm.segment_vessels(wt_img, bin_thresh = 5)
mutant_seg = vm.segment_vessels(mutant_img, bin_thresh = 1)

jacc_mt = vm.jaccard(mutant_label,mutant_seg)
jacc_wt = vm.jaccard(wt_label,wt_seg)

mt_hist = cv2.equalizeHist(mutant_img)
#plt.imshow(mt_hist, cmap=plt.get_cmap('binary_r'))

plt.figure()
plt.imshow(mutant_seg, cmap=plt.get_cmap('binary_r'))
plt.figure()
plt.imshow(wt_seg, cmap=plt.get_cmap('binary_r'))

plt.figure()
plt.imshow(mutant_img, cmap=plt.get_cmap('binary'))
plt.figure()
plt.imshow(wt_img, cmap=plt.get_cmap('binary_r'))

plt.figure()
plt.imshow(mutant_label, cmap=plt.get_cmap('binary_r'))
plt.figure()
plt.imshow(wt_label, cmap=plt.get_cmap('binary_r'))

mutant_label[mutant_label>0] = 1
overlap_mt = 2*mutant_label+mutant_seg
overlap_disp = overlap_mt*75
plt.figure(); plt.imshow(overlap_disp, cmap = plt.get_cmap('binary_r'))

wt_label[wt_label>0] = 1
overlap_wt = 2*wt_label+wt_seg
overlap_disp_wt = overlap_wt*75
plt.figure(); plt.imshow(overlap_disp_wt, cmap = plt.get_cmap('binary_r'))


###########################################################################
plexus_im = cv2.imread(data_path + 'parameter_calc/img.png',0)
plexus_label = cv2.imread(data_path +'parameter_calc/label.png',0)
true_skel_label = cv2.imread(data_path + 'parameter_calc/skel_label.png',0)

true_skel_label[true_skel_label>0] = 1
true_skel = skeletonize(true_skel_label)
kernel = np.ones((5,5),np.uint8)
true_skel_dilate = cv2.dilate(true_skel.astype(np.uint8),kernel)
overlay = draw.overlay_skeleton_2d(plexus_im,true_skel,dilate = 2)

edges, branchpoints = vm.find_branchpoints(true_skel)

plt.figure();
plt.imshow(true_skel, cmap = plt.get_cmap('binary_r'))
plt.figure();
plt.imshow(edges, cmap = plt.get_cmap('binary_r'))
plt.figure()
plt.imshow(branchpoints, cmap = plt.get_cmap('binary_r'))

num_labels, edge_labels = cv2.connectedComponents(edges, connectivity = 8)
plt.figure();
plt.imshow(edge_labels)

edge_num = 128
segment = np.zeros_like(edge_labels)
segment[edge_labels == 128] = 1

########################################################################
diameters, label, label_intensity = vm.vessel_diameter(label, segment)

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
median_index = np.where(sorted_distances == round(np.median(sorted_distances)))[0][0]
segment_median = sort_indexes[median_index]

start_pt = segment_indexes[sort_indexes[median_index-3]]
end_pt = segment_indexes[sort_indexes[median_index+3]]
median_pt = segment_indexes[sort_indexes[median_index]]
slope = (start_pt[0]-end_pt[0])/(start_pt[1]-end_pt[1])
cross_slope = -1/slope

cross_length = find_segment_crossline_length(label, median_pt,cross_slope)

diameters = []
for i in range(10,len(sort_indexes)-3,10):
    print(i)
    start_pt = segment_indexes[sort_indexes[i-3]]
    end_pt = segment_indexes[sort_indexes[i+3]]
    mid_pt = segment_indexes[sort_indexes[i]]
    slope = (start_pt[0]-end_pt[0])/(start_pt[1]-end_pt[1])

    if slope != 0:
        cross_slope = -1/slope

        x_dist, y_dist = distance_along_line(mid_pt, cross_slope, cross_length)
        cross_index = calculate_crossline(mid_pt, cross_slope, x_dist, y_dist)
    else:
        x_dist = 0
        y_dist = cross_length
        
        x1 = np.int(mid_pt[0]-x_dist)
        y1 = np.int(mid_pt[1]-y_dist)

        x2 = np.int(mid_pt[0]+x_dist)
        y2 = np.int(mid_pt[1]+y_dist)

        cross_index = list(bresenham(x1,y1,x2,y2))
    
    label_intensity = []
    for j in cross_index:
        label[j[0],j[1]] = 3
        label_intensity.append(label[j[0],j[1]])
    this_diameter = np.sum(label_intensity)
    diameters.append(this_diameter)





def crossline_endpoints(label,start,slope):    
    current_point = start
    current_label_val = label[current_point[0],current_point[1]]
    while current_label_val:
        current_x = current_point[0]-1
        current_y = current_point[1]-slope
        current_point = np.int(round(current_x)), np.int(round(current_y))
        current_label_val = label[current_point[0],current_point[1]]
    end_point1 = current_point
    
    current_point = start
    current_label_val = label[current_point[0],current_point[1]]
    while current_label_val:
        current_x = current_point[0]+1
        current_y = current_point[1]+slope
        current_point = np.int(round(current_x)), np.int(round(current_y))
        current_label_val = label[current_point[0],current_point[1]]
    end_point2 = current_point
    return end_point1, end_point2

def find_segment_crossline_length(label,start,slope):
    current_point = start
    current_label_val = label[current_point[0],current_point[1]]
    while current_label_val:
        current_x = current_point[0]-slope
        current_y = current_point[1]-1
        current_point = current_x, current_y
        current_point_discrete = np.int(np.round(current_point[0])), np.int(np.round(current_point[1]))
        print(current_y, current_point_discrete[1])
        current_label_val = label[current_point_discrete[0], current_point_discrete[1]]
    end_point = np.int(np.round(current_x)), np.int(np.round(current_y))
    vessel_radius = distance.chebyshev(end_point,start)
    cross_thickness = vessel_radius*1.5
    return cross_thickness

def find_next_point(point, slope):
    if np.abs(slope)<1 and np.abs(slope)>0:
        distance = np.abs(1/slope) + 1
    elif np.abs == 0:
        distance = 1
    else:
        distance = np.abs(slope)+1
    
    if slope<0:
        distance = distance*-1
    x_dist, y_dist = distance_along_line(point,slope,distance)
    return x_dist, y_dist


def distance_along_line(point,slope,distance):
    x_dist = np.sqrt(distance**2/(slope**2+1))
    y_dist = x_dist*slope
    
    x_dist = np.round(x_dist)
    y_dist = np.round(y_dist)
    
    return x_dist, y_dist
    
def calculate_crossline(point, slope, x_dist, y_dist):
    x1 = np.int(point[0]-x_dist)
    y1 = np.int(point[1]-y_dist)
    
    x2 = np.int(point[0]+x_dist)
    y2 = np.int(point[1]+y_dist)
    
    cross_index = list(bresenham(x1,y1,x2,y2))
    
    return cross_index

img_overlay = deepcopy(plexus_im)
for i in cross_index2:
    img_overlay[i[0], i[1]] = 255
plt.figure(); plt.imshow(img_overlay)
