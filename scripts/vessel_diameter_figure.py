#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:08:22 2022

vessel diameter figures

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

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/E3_diameter/'

im = cv2.imread(data_path+'emb7/img.png',0)
seg = vm.brain_seg(im, preprocess = True, filter = 'frangi', thresh = 10)
label = cv2.imread(data_path+'emb7/label.png',0)

skel, edges, bp = vm.skeletonize_vm(seg)
edge_count, edge_labels = cv2.connectedComponents(edges)

viz, diameters = whole_anatomy_diameter(im, seg, edge_labels, minimum_length = 25, pad_size = 50)
i = 52
all_diams, temp_diam, temp_viz = visualize_vessel_diameter(edge_labels, i, seg, im) 
