#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:29:54 2022

Diameter sandbox

@author: sean
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:30:13 2021

vm E3 diameter measurement - automatic

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from copy import deepcopy

data_path = '/home/sean/Documents/vm_manuscript/E3_diameter/'

test_embs = ['emb1', 'emb3', 'emb5', 'emb7', 'emb9']

im_list = []
label_list = []
edge_label_list = []
viz_list = []
seg_list = []
for i in test_embs:
    im = cv2.imread(data_path+i+'/img.png',0)
    seg = vm.brain_seg(im, filter = 'frangi', thresh = 20)
    skel = skeletonize(seg)
    edges, bp = vm.find_branchpoints(skel)
    _, edge_labels = cv2.connectedComponents(edges)
    viz, diameters = vm.whole_anatomy_diameter(im, seg, edge_labels)
    
    im_preproc = vm.contrast_stretch(im)
    im_preproc = vm.preprocess_seg(im_preproc)
    im_list.append(im)
    edge_label_list.append(edge_labels)
    viz_list.append(viz)
    seg_list.append(seg)

vm.overlay_segmentation(im_list[0], viz_list[0]+edge_label_list[0])
vm.overlay_segmentation(im_list[1], viz_list[1]+edge_label_list[1])
vm.overlay_segmentation(im_list[2], viz_list[2]+edge_label_list[2])
vm.overlay_segmentation(im_list[3], viz_list[3]+edge_label_list[3])
vm.overlay_segmentation(im_list[4], viz_list[4]+edge_label_list[4])

emb1_segs = [53, 55,47, 28, 20]
emb3_segs = [32,51,58,48,29]
emb5_segs = [36,46,51,20, 27]
emb7_segs = [60,64,38,37,31]
emb9_segs =[7,56,61,45,33]

E1_mean_diameter = []
E1_viz = []
for i in emb1_segs:
    im = im_list[0]
    im = vm.preprocess_seg(im)
    im = vm.contrast_stretch(im)
    temp_diam, temp_mean, temp_viz = vm.visualize_vessel_diameter(edge_label_list[0], i, seg_list[0],im)
    E1_mean_diameter.append(temp_mean)
    E1_viz.append(temp_viz)
    
    
E3_mean_diameter = []
E3_viz = []
for i in emb3_segs:
    im = im_list[1]
    im = vm.preprocess_seg(im)
    im = vm.contrast_stretch(im)
    temp_diam, temp_mean, temp_viz = vm.visualize_vessel_diameter(edge_label_list[1], i, seg_list[1],im)
    E3_mean_diameter.append(temp_mean)
    E3_viz.append(temp_viz)

    
E5_mean_diameter = []
E5_viz = []
for i in emb5_segs:
    im = im_list[2]
    im = vm.preprocess_seg(im)
    im = vm.contrast_stretch(im)
    temp_diam, temp_mean, temp_viz = vm.visualize_vessel_diameter(edge_label_list[2], i, seg_list[2], im)
    E5_mean_diameter.append(temp_mean)
    E5_viz.append(temp_viz)

E7_mean_diameter = []
E7_viz = []
for i in emb7_segs:
    im = im_list[3]
    im = vm.preprocess_seg(im)
    im = vm.contrast_stretch(im)
    temp_diam, temp_mean, temp_viz = vm.visualize_vessel_diameter(edge_label_list[3], i, seg_list[3],im)
    E7_mean_diameter.append(temp_mean)
    E7_viz.append(temp_viz)
    
E9_mean_diameter = []
E9_viz = []
for i in emb9_segs:
    im = im_list[4]
    im = vm.preprocess_seg(im)
    im = vm.contrast_stretch(im)
    temp_diam, temp_mean, temp_viz = vm.visualize_vessel_diameter(edge_label_list[4], i, seg_list[4],im)
    E9_mean_diameter.append(temp_mean)
    E9_viz.append(temp_viz)
    
vm.overlay_segmentation(im_list[4], seg_list[4]+E9_viz[0])



###################################################################
edge_labels = edge_label_list[3]
seg = seg_list[3]
im = im_preproc_list[3]
use_label = False
segment_number = 60

def visualize_vessel_diameter(edge_labels, segment_number, seg, im, use_label = False):
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==segment_number] = 1
    segment_median = vm.segment_midpoint(segment)

    vx,vy = vm.tangent_slope(segment, segment_median)
    bx,by = vm.crossline_slope(vx,vy)
    
    viz = np.zeros_like(seg)
    cross_length = vm.find_crossline_length(bx,by, segment_median, seg)
    
    if cross_length == 0:
        diameter = 0
        mean_diameter = 0
        return diameter, mean_diameter, viz
    
    diameter = []
    segment_inds = np.argwhere(segment)
    for i in range(10,len(segment_inds),10):
        print(i)
        this_point = segment_inds[i]
        vx,vy = vm.tangent_slope(segment, this_point)
        bx,by = vm.crossline_slope(vx,vy)
        _, cross_index = vm.make_crossline(bx,by, this_point, cross_length)
        if use_label:
            cross_vals = vm.crossline_intensity(cross_index,seg)
            diam = vm.label_diameter(cross_vals)
        else:
            cross_vals = vm.crossline_intensity(cross_index, im)
            diam = vm.fwhm_diameter(cross_vals)
        if diam == 0:
            val = 5
        else:
            val = 10
        for ind in cross_index:
            viz[ind[0], ind[1]] = val
        diameter.append(diam)
    diameter = [x for x in diameter if x != 0]
    if diameter:
        mean_diameter = np.mean(diameter)
    else:
        mean_diameter = 0
    
    return diameter, mean_diameter, viz


