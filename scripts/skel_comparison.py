#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:09:02 2021

compares skeleton from true annotation to skeleton drawn by hand

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skan import draw
from skan import skeleton_to_csgraph
import vessel_metrics as vm
from skimage import draw

data_dir = '/home/sean/Documents/Calgary_postdoc/Data/czi_projections/Channel0/contrast_enhanced/Feb_23_v2/'
label = cv2.imread(data_dir + 'label.png',0)
image = cv2.imread(data_dir + 'img_enhanced.png',0)

label[label>0] = 1
skel, label = vm.fill_holes(label, 50)

skel_label = cv2.imread(data_dir + 'skel_mask/label.png',0)
skel_label[skel_label>0] = 1
true_skel = skeletonize(skel_label)

kernel = np.ones((5,5),np.uint8)
skel_dilate = cv2.dilate(skel.astype(np.uint8),kernel)
true_skel_dilate = cv2.dilate(true_skel.astype(np.uint8),kernel)

overlap = skel_dilate + true_skel_dilate
plt.figure(); plt.imshow(overlap)

plt.figure(); 
overlay = draw.overlay_skeleton_2d(image,skel,dilate = 2)