#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:28:42 2021

Vessel diameter test

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

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_112620/'
file_name = 'flk gata inx 30hpf Feb 23 2019 E4 good qual.czi'

volume = vm.preprocess_czi(data_path, file_name)
reslice_volume = vm.sliding_window(volume,4)

test_slice = reslice_volume[30,:,:]
label = vm.segment_vessels(test_slice)

skel = skeletonize(label)
edges = vm.find_branchpoints(skel)

num_labels, edge_labels = cv2.connectedComponents(edges, connectivity = 8)

edge_labels, edges = vm.remove_small_segments(edge_labels, 50)

segment_number = 63
this_segment = np.zeros_like(edges)
this_segment[edge_labels == segment_number] = 1

segment_index = np.where(this_segment)
test_segment = np.zeros_like(edges)
for i in range(len(segment_index[0])):
    test_segment[segment_index[0][i],segment_index[1][i]] = i+1

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

#for i in range(10,len(sorted_distances),10):
#    start_pt = segment_index[0][sort_indexes[i-3]], segment_index[1][sort_indexes[i-3]]
#    end_pt = segment_index[0][sort_indexes[i+3]], segment_index[1][sort_indexes[i+3]]
#    slope = (start_pt[0]-end_pt[0])/(start_pt[1]-end_pt[1])
#    perp_slope = -1/slope
#    mid_point = segment_index[0][sort_indexes[i]], segment_index[1][sort_indexes[i]]
#    
#    perp_end = 
    
i = 10
start_pt = segment_index[0][sort_indexes[i-3]], segment_index[1][sort_indexes[i-3]]
end_pt = segment_index[0][sort_indexes[i+3]], segment_index[1][sort_indexes[i+3]]
slope = (start_pt[0]-end_pt[0])/(start_pt[1]-end_pt[1])
perp_slope = -1/slope
mid_point = segment_index[0][sort_indexes[i]], segment_index[1][sort_indexes[i]]

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
        current_x = current_point[0]-1
        current_y = current_point[1]-slope
        current_point = current_x, current_y
        current_label_val = label[np.int(np.round(current_point[0])),np.int(np.round(current_point[1]))]
    end_point = np.int(np.round(current_x)), np.int(np.round(current_y))
    vessel_radius = distance.chebyshev(end_point,start)
    cross_thickness = vessel_radius*1.5
    return cross_thickness

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
    
end1, end2 = crossline_endpoints(label,mid_point,perp_slope)
label_test = label
cross_index = list(bresenham(end1[0], end1[1], end2[0],end2[1]))
for i in cross_index:
    label_test[i[0], i[1]] = 5
    
vessel_radius1 = distance.euclidean(end1,mid_point)
vessel_radius2 = distance.euclidean(end2, mid_point)

cross_thickness = vessel_radius1*1.5

dx, dy = distance_along_line(mid_point, perp_slope, cross_thickness)
cross_index = calculate_crossline(mid_point, perp_slope, dx, dy)

label_test = label.copy()
img_slice = test_slice.copy()
for i in cross_index:
    label_test[i[0],i[1]] = 3
    img_slice[i[0],i[1]] = 255
    
img_intensity = []
label_intensity = []
for i in cross_index:
    img_intensity.append(test_slice[i[0], i[1]])
    label_intensity.append(label[i[0],i[1]])

plt.figure(); plt.plot(range(len(img_intensity)),img_intensity)
plt.figure(); plt.plot(range(len(label_intensity)),label_intensity)    

def vessel_diameter(segment_indexes, label, segment):
    segment_endpoints = find_endpoints(segment)
    endpoint_index = np.where(segment_endpoints)
    first_endpoint = endpoint_index[0][0], endpoint_index[1][0]

    distances = []
    for i in range(len(segment_index[0])):
        this_pt = segment_index[0][i], segment_index[1][i]
        distances.append(distance.chebyshev(first_endpoint, this_pt))
    sort_indexes = np.argsort(distances)
    sorted_distances = sorted(distances)
    median_index = np.where(sorted_distances == np.median(sorted_distances))[0][0]
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
    return diameters, label
            

cross_index = calculate_crossline(median_pt, cross_slope, x_dist, y_dist)

label_test = label.copy()
for i in cross_index:
    label_test[i[0],i[1]] = 3
    
    
diameters, label_out = vessel_diameter(segment_indexes, label, segment)
plt.figure(); plt.plot(range(len(diameters)), diameters)
plt.figure(); plt.imshow(label_out)

display_flag = True
if display_flag:
    plt.figure(); plt.imshow(test_slice)
    plt.figure(); plt.imshow(label)
    plt.figure(); plt.imshow(skel)
    plt.figure(); plt.imshow(edges)
    plt.figure(); plt.imshow(edge_labels)
