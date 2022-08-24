#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:48:41 2021

skeleton evolution algorithm test

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
from skimage.morphology import skeletonize
from skimage.morphology import medial_axis
import skeleton_evolution as sk
from scipy.spatial import distance
from bresenham import bresenham
from copy import deepcopy

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/'
out_path = '/home/sean/Documents/Calgary_postdoc/Data/abstract_42120/'

dataset = ['img1_14_json/', 'img1_22_json/', 'img2_20_json/', 'img3_12_json/', 'img4_12_json/', 'img4_18_json/']

img_list = []; label_list = []; seg_list = []
for file in dataset:
    img = cv2.imread(out_path + file + 'img.png',0)
    label = cv2.imread(out_path + file + 'label.png',0)
    
    seg = vm.segment_vessels(img)
    img_list.append(img)
    label = np.uint8(label)
    label_list.append(label)
    seg = np.uint8(seg)
    seg_list.append(seg)
 
test_label = deepcopy(label_list[0])
plt.figure(); plt.imshow(test_label)

skel1 = skeletonize(test_label)
med = medial_axis(test_label)

skel_dse3 = sk.DSE_v3(test_label, 20)


