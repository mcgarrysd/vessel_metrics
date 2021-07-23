#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 08:40:43 2021

Branchpoint density test

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

dataset = ['img1_14_json/', 'img1_22_json/', 'img2_20_json/', 'img3_12_json/', 'img4_12_json/', 'img4_18_json/']
out_path = '/home/sean/Documents/Calgary_postdoc/Data/abstract_42120/'

jacc = []
img_list = []
label_list = []
seg_list = []
for file in dataset:
    img = cv2.imread(out_path + file + 'img.png',0)
    label = cv2.imread(out_path + file + 'label.png',0)
    
    seg = vm.segment_vessels(img)
    
    jacc.append(vm.jaccard(label,seg))
    img_list.append(img)
        
    label_binary = np.uint8(label)
    seg_binary = np.uint8(seg)
    
    label_list.append(label_binary)
    seg_list.append(seg_binary)

test_img = deepcopy(img_list[3])
test_label = deepcopy(label_list[3])
test_seg = deepcopy(seg_list[3])

test_label[test_label>0] = 1
test_seg[test_seg>1] = 1

l_skel = skeletonize(test_label)
l_edges, l_bp = vm.find_branchpoints(l_skel)
_, l_edge_labels = cv2.connectedComponents(l_edges, connectivity = 8)
l_edge_labels, l_edges = vm.remove_small_segments(l_edge_labels, 50)

s_skel = skeletonize(test_seg)
s_edges, s_bp = vm.find_branchpoints(s_skel)
_, s_edge_labels = cv2.connectedComponents(s_edges, connectivity = 8)
s_edge_labels, s_edges = vm.remove_small_segments(s_edge_labels, 50)

def fix_disconnected_segments(edges,branch_points):
    pseudo_skel = edges+branch_points
    pseudo_skel[pseudo_skel>0] = 1
    
    # remove terminal segment endpoints
    
    new_edges, new_bp = vm.find_branchpoints(pseudo_skel)
    
    return new_edges, new_bp

def remove_terminal_endpoints(skel):
    skel[skel>0]=1
    skel_index = np.argwhere(skel == True)
    tile_sum=[]
    neighborhood_image = np.zeros(skel.shape)

    for i,j in skel_index:
        this_tile = skel[i-1:i+2,j-1:j+2]
        tile_sum.append(np.sum(this_tile))
        neighborhood_image[i,j] = np.sum(this_tile)
    terminal_points = np.zeros_like(neighborhood_image)
    terminal_points[neighborhood_image == 2] = 1
    
    out_skel = np.zeros_like(skel)
    out_skel[skel == 1] = 1
    out_skel[terminal_points == 1] = 0
    
    return out_skel, terminal_points

def remove_skeleton_burrs(skel):
    skel[skel>0]=1
    skel_index = np.argwhere(skel == True)
    tile_sum=[]
    neighborhood_image = np.zeros(skel.shape)

    for i,j in skel_index:
        this_tile = skel[i-1:i+2,j-1:j+2]
        tile_sum.append(np.sum(this_tile))
        neighborhood_image[i,j] = np.sum(this_tile)
    burrs = np.zeros_like(neighborhood_image)
    burrs[neighborhood_image == 4] = 1
    
    out_skel = np.zeros_like(skel)
    out_skel[skel == 1] = 1
    out_skel[burrs == 1] = 0
    
    return out_skel, burrs

def branchpoint_density(skel, label):
    _, bp = vm.find_branchpoints(skel)
    _, bp_labels = cv2.connectedComponents(bp, connectivity = 8)
    
    skel_inds = np.argwhere(skel > 0)
    
    bp_density = []
    for i in range(0,len(skel_inds), 50):
        x = skel_inds[i][0]; y = skel_inds[i][1]
        this_tile = bp_labels[x-25:x+25,y-25:y+25]
        bp_number = len(np.unique(this_tile))-1
        bp_density.append(bp_number)
        
    bp_density = np.array(bp_density)
    bp_density[bp_density<0] = 0
    return bp_density

def plexus_complexity(label, min_val, max_val):
    label[label>0] = 1
    label_inv = np.bitwise_not(label)
    label_inv[label_inv<255] = 0
    _, inverted_labels, stats, _ = cv2.connectedComponentsWithStats(label_inv)
    unique_labels = np.unique(inverted_labels)
    num_label_pixels = len(np.argwhere(label == 1))
    hole_sizes = []
    for i in unique_labels:
        numel = len(np.argwhere(inverted_labels == i))
        if numel>max_val:
            inverted_labels[inverted_labels == i] = 0
        elif numel<min_val:
            inverted_labels[inverted_labels == i] = 0
        else:
            hole_sizes.append(numel)
    num_hole_pixels = len(np.argwhere(inverted_labels>0))
    percent_holes = num_hole_pixels/num_label_pixels*100
    num_holes = len(np.unique(inverted_labels))-1
    mean_hole_size = np.mean(hole_sizes)
    return percent_holes, num_holes, mean_hole_size