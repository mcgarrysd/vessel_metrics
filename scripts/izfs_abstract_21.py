#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:40:03 2021

Data for abstract submitted to IZFS 04/21/21

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

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/'
data_files = ['35M-59H inx 48hpf Apr 14 2019 E2.czi', '35M-59H inx 48hpf Apr 14 2019 E9.czi',\
'flk gata 48hpf Jul 26 2019 E5.czi', 'flk gata inx 48hpf Apr 14 2019 E4.czi']
out_path = '/home/sean/Documents/Calgary_postdoc/Data/abstract_42120/'

img_slices = [[14, 22], [15, 20], [6, 12], [12, 18]]
img_list = list(zip(data_files, img_slices))

create_images = False
if create_images:
    count = 0
    for i in img_list:
        vol = vm.preprocess_czi(data_path, i[0])
        vol = vm.sliding_window(vol, 4)
        count+=1
        for j in i[1]:
            file_name = 'img' + str(count) + '_' + str(j) + '.png'
            img = vol[j,:,:]
            cv2.imwrite(out_path + file_name, img)
            

dataset = ['img1_14_json/', 'img1_22_json/', 'img2_20_json/', 'img3_12_json/', 'img4_12_json/', 'img4_18_json/']
jacc = []
img_list = []
label_list = []
seg_list = []
conn_list = []; area_list = []; length_list = [];
for file in dataset:
    img = cv2.imread(out_path + file + 'img.png',0)
    label = cv2.imread(out_path + file + 'label.png',0)
    
    seg = vm.segment_vessels(img)
    
    jacc.append(vm.jaccard(label,seg))
    img_list.append(img)
    label_list.append(label)
    seg_list.append(seg)
    
    label_binary = np.uint8(label)
    seg_binary = np.uint8(seg)
    
    connectivity, area, length = vm.cal(label_binary,seg_binary)
    conn_list.append(connectivity)
    area_list.append(area)
    length_list.append(length)

mean_jacc = np.mean(jacc)
mean_wt = np.mean(jacc[0:2])
mean_mt = np.mean(jacc[3:])

plt.figure(); plt.imshow(label_list[3])
plt.figure(); plt.imshow(seg_list[3])

label = seg_list[3]
label_orig = deepcopy(label)
skel = skeletonize(label)
edges = vm.find_branchpoints(skel)

num_labels, edge_labels = cv2.connectedComponents(edges, connectivity = 8)

edge_labels, edges = vm.remove_small_segments(edge_labels, 50)

segment_number = 18, 11, 12 

for i in segment_number:
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==i] = 1
    diameters, label_out2, label_intensity = vessel_diameter(label_orig, segment)

    
    
    
    
###############################################################################
    
def vessel_diameter(label, segment):
    segment_endpoints = vm.find_endpoints(segment)
    endpoint_index = np.argwhere(segment_endpoints)
    first_endpoint = endpoint_index[0][0], endpoint_index[0][1]
    segment_indexes = np.argwhere(segment==1)
    
    distances = []
    for i in range(len(segment_indexes)):
        this_pt = segment_indexes[i][0], segment_indexes[i][1]
        distances.append(distance.chebyshev(first_endpoint, this_pt))
    sort_indexes = np.argsort(distances)
    sorted_distances = sorted(distances)
    median_index = np.int(len(sort_indexes)/2)
    segment_median = sort_indexes[median_index]
    
    start_pt = segment_indexes[sort_indexes[median_index-5]]
    end_pt = segment_indexes[sort_indexes[median_index+5]]
    median_pt = segment_indexes[sort_indexes[median_index]]
    slope = (start_pt[0]-end_pt[0])/(start_pt[1]-end_pt[1])
    cross_slope = -1/slope

    cross_length = find_segment_crossline_length(label, median_pt,cross_slope)
    
    diameters = []
    for i in range(10,len(sort_indexes)-5,10):
        print(i)
        start_pt = segment_indexes[sort_indexes[i-5]]
        end_pt = segment_indexes[sort_indexes[i+5]]
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
            label_intensity.append(label[j[0],j[1]])
            label[j[0],j[1]] = 3
        this_diameter = np.sum(label_intensity)
        diameters.append(this_diameter)
    return diameters, label, label_intensity


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


