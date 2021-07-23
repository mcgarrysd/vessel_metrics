#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 06:44:58 2021

label_segments_small

does a small chunk of an image, hopefully in time for lab meeting

@author: sean
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:03:40 2020

label_segments
Accepts binary mask as input, outputs binary mask with every segment individually 
labelled

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

data_dir = '/home/sean/Documents/Calgary_postdoc/Data/czi_projections/Channel0/contrast_enhanced/Feb_23_E3/'

label = cv2.imread(data_dir + 'label.png',0)
image = cv2.imread(data_dir + 'img_enhanced.png',0)

label_binary = np.zeros_like(label)
label_binary[label>0] = 1
skel = skeletonize(label_binary)

branch_points, edges = find_branchpoints(skel)
end_points = find_endpoints(edges)

num_labels, edge_labels = cv2.connectedComponents(edges, connectivity = 8)

small_im = image[300:599,300:599]
small_mask = label_binary[300:599,300:599]
small_edge_label = edge_labels[300:599,300:599]
unique_seg_small, all_seg_small = vessel_length(small_edge_label)
small_edge_label = edge_labels[300:599,300:599]

unique_segments_sm, segment_counts_sm = vessel_length(small_edge_label)

sm_length_map = pixelwise_vessel_length(unique_segments_sm, segment_counts_sm, small_mask, small_edge_label)

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
    segments_counts = segment_counts[1:]
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
        
        lookup_table_ind = np.argwhere(unique_segments == closest_segment)
        length_map[i,j] = all_segment_lengths[lookup_table_ind]
    return length_map

    
