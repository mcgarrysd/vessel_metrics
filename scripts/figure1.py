#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 20:12:58 2022

figure 1

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


data_path2 = '/media/sean/SP PHD U3/from_home/vm_manuscript/raw_data/Wnt Treatment/'
g = 'Nov14'
data_files = os.listdir(data_path2+g+'/')
data_files = [i for i in data_files if 'DMSO' in i]
file = data_files[0]

volume = vm.preprocess_czi(data_path2+g+'/',file, channel = 1)
slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
this_slice = reslice[0]
slice_num = slice_thickness/2
single_slice = volume[slice_num.astype(np.uint8)]
preproc = vm.preprocess_seg(this_slice.astype(np.uint8))

enhanced_im = meijering(preproc, sigmas = range(1,8,2), mode = 'reflect', black_ridges = False)

seg = vm.brain_seg(this_slice.astype(np.uint8), filter = 'meijering', thresh = 40)
skel, edges, bp = vm.skeletonize_vm(seg)
_, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))

dens, density_array = vm.vessel_density(this_slice, seg, 16, 16)

segment_number = 21
diam, mean_diam, viz = visualize_vessel_diameter(edge_labels, segment_number, seg, this_slice, use_label = False)
viz_adjusted = viz*10
viz_adjusted[0,0] = 1

plt.close('all')
plt.figure(); plt.imshow(single_slice, cmap = 'gray')
plt.figure(); plt.imshow(this_slice, cmap = 'gray')
plt.figure(); plt.imshow(preproc, cmap = 'gray')
plt.figure(); plt.imshow(enhanced_im, cmap = 'gray')
vm.overlay_segmentation(this_slice, seg)

vm.overlay_segmentation(this_slice, seg+skel*2)
vm.overlay_segmentation(this_slice, edge_labels+bp)


vm.overlay_segmentation(this_slice, dens, alpha = 0.3)
vm.overlay_segmentation(this_slice, viz_adjusted)
