#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:03:40 2020

label_segments

Sandbox for the development of the vessel_metrics module 

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skan import draw
from skan import skeleton_to_csgraph
from scipy.spatial import distance

plt.close('all')
display_flag = False
map_flag = True
small_segment_flag = False

data_dir = '/home/sean/Documents/Calgary_postdoc/Data/czi_projections/Channel0/contrast_enhanced/Feb_23_E3/'

label = cv2.imread(data_dir + 'label.png',0)
image = cv2.imread(data_dir + 'img_enhanced.png',0)

label_binary = np.zeros_like(label)
label_binary[label>0] = 1
skel = skeletonize(label_binary)


if display_flag:
    plt.figure(); 
    plt.subplot(1,2,1); plt.imshow(label)
    plt.subplot(1,2,2); plt.imshow(skel)
    plt.figure(); 
    overlay = draw.overlay_skeleton_2d(image,skel,dilate = 2)

skel_overlay = skel*255+image

branch_points, edges = find_branchpoints(skel)
end_points = find_endpoints(edges)

num_labels, edge_labels = cv2.connectedComponents(edges, connectivity = 8)


unique_segments, segment_counts = vessel_length(edge_labels)


# FILL HOLES IN MASK ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
label_inv = np.bitwise_not(label_binary)
label_inv[label_inv<255] = 0
_, inverted_labels, stats, _ = cv2.connectedComponentsWithStats(label_inv)

hole_size = 50
vessel_sizes = stats[:,4]
small_vessel_inds = np.argwhere(vessel_sizes<hole_size)

for v in small_vessel_inds:
    inverted_labels[inverted_labels == v] = 0
    
label_mask = np.zeros_like(inverted_labels_mask)
label_mask[inverted_labels==0] = 1 

new_skel = skeletonize(label_mask)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

branch_points, edges = find_branchpoints(new_skel)
end_points = find_endpoints(edges)
num_labels, edge_labels = cv2.connectedComponents(edges, connectivity = 8)
tort = tortuosity(edge_labels,end_points)

def tortuosity(edge_labels, end_points):
    endpoint_labeled = edge_labels*end_points
    unique_labels = np.unique(edge_labels)
    tortuosity = []
    for u in unique_labels:
        this_segment = np.zeros_like(edge_labels)
        this_segment[edge_labels == u] = 1
        
        these_endpoints = np.argwhere(endpoint_labeled == u)
        try:
            endpoint_distance = distance.euclidean(these_endpoints[0], these_endpoints[1])
        except:
            print(u)
        segment_length = np.sum(this_segment)
        tortuosity.append(endpoint_distance/segment_length)
    return tortuosity

def display_small_segments(edge_labels, unique_segments, segment_counts,skel_overlay,label):
    plt.close('all')
    short_seg_inds = np.argwhere(segment_counts<10)
    short_seg_labels = unique_segments[short_seg_inds]
    short_seg_labels = np.concatenate(short_seg_labels, axis = 0)
    count = 0
    for this_seg in short_seg_labels:
        this_seg_map = np.zeros_like(edge_labels)
        this_seg_map[edge_labels == this_seg] = 1
        this_seg_inds = np.argwhere(this_seg_map == 1)
        
        if count<10:
            x_ind, y_ind = this_seg_inds[0]
            tile = edge_labels[x_ind-15:x_ind+15, y_ind-15:y_ind+15]
            tile_label = this_seg_map[x_ind-15:x_ind+15, y_ind-15:y_ind+15]
            tile_overlay = skel_overlay[x_ind-50:x_ind+50, y_ind-50:y_ind+50]
            tile_mask = label[x_ind-50:x_ind+50, y_ind-50:y_ind+50]
            plt.figure(); 
            plt.subplot(2,2,1); plt.imshow(tile_label)
            plt.subplot(2,2,2); plt.imshow(tile)
            plt.subplot(2,2,3); plt.imshow(tile_overlay)
            plt.subplot(2,2,4); plt.imshow(tile_mask)
            plt.savefig('/home/sean/Documents/Calgary_postdoc/skel_images_011321/segment_'+str(count)+'.png')
        count = count+1

if small_segment_flag:
    display_small_segments(edge_labels, unique_segments, segment_counts,skel_overlay, label)

def replace_small_segments(edge_labels, unique_segments, segment_counts):
    short_seg_inds = np.argwhere(segment_counts<10)
    short_seg_labels = unique_segments[short_seg_inds]
    
    for this_seg in short_seg_labels:
        this_seg_map = np.zeros_like(edge_labels)
        this_seg_map[edge_labels == this_seg] = 1
        this_seg_inds = np.argwhere(this_seg_map == 1)
        
        x_ind, y_ind = this_seg_inds[0]
        tile = edge_labels[x_ind-10:x_ind+10, y_ind-10:y_ind+10]
        
def remove_small_segments(edge_labels):
    unique_labels = np.unique(edge_labels)
    unique_labels = unique_labels[1:]
    for u in unique_labels:
        this_seg_count = np.shape(np.argwhere(edge_labels==u))[0]
        if this_seg_count < 5:
            edge_labels[edge_labels==u] = 0
    temp_edge_labels = np.zeros_like(edge_labels)
    temp_edge_labels[edge_labels>0] = 1
    temp_edge_labels = temp_edge_labels.astype(np.uint8)
    _, edge_labels_new = cv2.connectedComponents(temp_edge_labels, connectivity = 8)
    return edge_labels_new, temp_edge_labels
    

def find_branchpoints(skel):
    skel_binary = np.zeros_like(skel)
    skel_binary[skel>0] = 1
    skel_index = np.argwhere(skel_binary == True)
    tile_sum=[]
    neighborhood_image = np.zeros(skel.shape)
    for i,j in skel_index:
        this_tile = skel_binary[i-1:i+2,j-1:j+2]
        tile_sum.append(np.sum(this_tile))
        neighborhood_image[i,j] = np.sum(this_tile)
    branch_points_messy = np.zeros_like(neighborhood_image)
    branch_points_messy[neighborhood_image>3] = 1
    branch_points_messy = branch_points_messy.astype(np.uint8)

    edges = np.zeros_like(branch_points_messy)
    edges = skel.astype(np.uint8) - branch_points_messy
    
    return branch_points_messy, edges

def find_endpoints(edges):
    edge_binary = np.zeros_like(edges)
    edge_binary[edges>0] = 1
    edge_index = np.argwhere(edge_binary == True)
    tile_sum=[]
    neighborhood_image = np.zeros(edges.shape)

    for i,j in edge_index:
        this_tile = edge_binary[i-1:i+2,j-1:j+2]
        tile_sum.append(np.sum(this_tile))
        neighborhood_image[i,j] = np.sum(this_tile)
    end_points = np.zeros_like(neighborhood_image)
    end_points[neighborhood_image == 2] = 1
    return end_points

def vessel_length(edge_labels):
    unique_segments, segment_counts = np.unique(edge_labels, return_counts = True)
    unique_segments = unique_segments[1:]
    segment_counts = segment_counts[1:]
    return unique_segments, segment_counts
    
def pixelwise_vessel_length(unique_segments,all_segment_lengths,label_binary,edge_labels):
    length_map = np.zeros(np.shape(edge_labels))
    mask_inds = np.argwhere(label_binary==1)
    count = 0
    for i,j in mask_inds:
        count+=1
        if count % 1000 == 0:
            print('comparison ' + str(count) + ' of ' + str(np.shape(mask_inds)[0]) )
        all_distances = []
        all_segments = np.empty([0,0])
        for u in unique_segments:
            segment_inds = np.argwhere(edge_labels == u)
            this_segment_label = np.ones([np.shape(segment_inds)[0],1])*u
            for x,y in segment_inds:
                all_distances.append(distance.euclidean([x,y],[i,j]))
            all_segments = np.append(all_segments,this_segment_label)
        min_distance = np.amin(all_distances)
        closest_segment = all_segments[np.argmin(all_distances)]
        length_map[i,j] = closest_segment
    return length_map

    
    
    






#    connectivity = 8
#    num_labels, stat_labels, stats, centroids = cv2.connectedComponentsWithStats(branch_points_messy,connectivity, cv2.CV_32S)
#    centroid_int = np.round(centroids).astype(np.uint16)
#    branch_points = np.zeros_like(branch_points_messy)
#    for i,j in centroid_int:
#        branch_points[j,i] = 1
#    

#    