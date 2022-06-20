#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 08:29:10 2022

github_read_me

sample code to generate images for the repo readme file

@author: sean
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import distance
from skimage import draw
import matplotlib.pyplot as plt
import os
from czifile import CziFile
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.draw import line
from bresenham import bresenham
import itertools
from math import dist

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/raw_data/DMH410uM75hpf/Jan15/'
data_list = os.listdir(data_path)
file = data_list[3]

volume = vm.preprocess_czi(data_path,file, channel = 0)
slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
vessel_raw = reslice[0]

vessel_preproc_disp = vm.preprocess_seg(vessel_raw)
vessel_preproc_disp = vm.contrast_stretch(vessel_preproc_disp)

median_size = 7
image = cv2.medianBlur(vessel_raw.astype(np.uint8),median_size)
ball_size = 400
vessel_preproc, background = subtract_background_rolling_ball(image, ball_size, light_background=False,
                                                            use_paraboloid=False, do_presmooth=True)
vessel_preproc = vm.contrast_stretch(vessel_preproc)

# filter options meijering, sato, frangi, jerman
# hole size (default 50) controls the size of small holes to be filled in post segmentation
# ditzle size (default 500) removes small objects post segmetnation
# sigmas (default (1,10,2)) controls sigma values input to enhancement filter
# thresh (default 60) is the threshold value to binarize the final enhanced image
vessel_seg = vm.brain_seg(vessel_raw, filter = 'frangi', thresh = 10)


skel, edges, bp = vm.skeletonize_vm(vessel_seg)
_, edge_labels = cv2.connectedComponents(edges)


# viz is an image of equal size to vessel_preproc containing binary vessel crosslines for visualization purposes
viz, diameters = vm.whole_anatomy_diameter(vessel_preproc, vessel_seg, edge_labels)

# to visualize a single segment use visualize vessel diameter
# diam_list is the diameter measured at each crossline
# mean_diam is the mean of diam_list
# segment_viz is a binary image showing crosslines for that segment
segment_number = 100
diam_list, mean_diam, segment_viz = vm.visualize_vessel_diameter(edge_labels, segment_number, vessel_seg,vessel_preproc)
    

# network length is the summation of all segment lengths
net_length = vm.network_length(edges)

# vessel density is the number of vessel pixels vs total pixels
# 16,16 denotes how many x and y chunks to break the image into (in this case 16 and 16)
_, vessel_density = vm.vessel_density(vessel_preproc, vessel_seg, 16, 16)

bp_density = vm.branchpoint_density(skel, vessel_seg)

# length is a list containing the segment length for every segment in edge_labels
_, length = vm.vessel_length(edge_labels)

_, end_points = vm.find_endpoints(edges)
tort, _ = vm.tortuosity(edge_labels, end_points)

############################################################
# pericyte
volume = vm.preprocess_czi(data_path,file, channel = 1)
slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
pericyte_raw = reslice[0]

peri_seg = np.zeros_like(pericyte_raw)
high_vals = np.zeros_like(pericyte_raw)
high_vals[pericyte_raw>75] = 1
peri_seg[(high_vals>0) & (vessel_seg>0)]=1

kernel = np.ones((3,3),np.uint8)
peri_seg = cv2.morphologyEx(peri_seg, cv2.MORPH_OPEN, kernel)

num_labels, labels = cv2.connectedComponents(peri_seg.astype(np.uint8))

unique_labels = np.array(np.nonzero(np.unique(labels))).flatten()

reduced_label = np.zeros_like(peri_seg)
for u in unique_labels:
    numel = len(np.argwhere(labels == u))
    if numel>15 and numel<500:
        reduced_label[labels == u] = 1
        
#############################################################
# Panels 
plt.close('all')

plt.figure(); 
plt.imshow(vessel_raw, cmap = 'gray')

plt.figure();
plt.imshow(vessel_preproc_disp, cmap = 'gray')

vm.overlay_segmentation(vessel_raw, vessel_seg)

vm.overlay_segmentation(vessel_raw, edge_labels)

segment_number = 78
diam_list, mean_diam, segment_viz = vm.visualize_vessel_diameter(edge_labels, segment_number, vessel_seg,vessel_raw)
segment_viz[0,0] = 1
vm.overlay_segmentation(vessel_raw, segment_viz)

density_ovl, vessel_density = vm.vessel_density(vessel_preproc, vessel_seg, 16, 16)
vm.overlay_segmentation(vessel_raw, density_ovl, alpha = 0.2)

vm.overlay_segmentation(vessel_raw, vessel_seg+viz)
