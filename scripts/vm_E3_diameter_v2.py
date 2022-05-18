#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:25:10 2022
vm_E3_diameter_v2

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

generate_data = True
data_path = '/media/sean/0012-D687/from_home/vm_manuscript/E1_combined/'
data_list = os.listdir(data_path)
data_list = data_list[0:5]

output_path = '/media/sean/0012-D687/from_home/vm_manuscript/diameter_segments/'

E1_segs = [116, 122, 77, 51, 34, 7, 18, 26, 148, 145]


for file in data_list:
    im = cv2.imread(data_path+file+'/img.png',0)
    seg = vm.brain_seg(im, filter = 'frangi', thresh = 20)
    skel = skeletonize(seg)
    edges, bp, new_skel = vm.prune_terminal_segments(skel)
    fixed_edges, fixed_bp = vm.fix_skel_artefacts(skel)
    edges, bp = vm.find_branchpoints(skel)
    _, edge_labels = cv2.connectedComponents(edges)
    vm.overlay_segmentation(im, edge_labels)