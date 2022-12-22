#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:24:11 2022

removed measurements

@author: sean
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import distance # Diameter measurement
import matplotlib.pyplot as plt
import os
from skimage.filters import meijering, hessian, frangi, sato
from skimage.draw import line # just in tortuosity
from bresenham import bresenham # diameter 
from skimage.util import invert
from skimage.filters.ridges import compute_hessian_eigenvalues
import itertools # fixing skeleton
from math import dist
from aicsimageio import AICSImage
from skimage import data, restoration, util # deprecated preproc
import timeit
from skimage.morphology import white_tophat, black_tophat, disk
import vessel_metrics as vm


data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/E1_segmentation/'

label_list = []
im_list = []
data_files = os.listdir(data_path)

for file in data_files:
    label_list.append(cv2.imread(data_path+file+'/label.png',0))
    im_list.append(cv2.imread(data_path+file+'/img.png',0))
    
test_im = im_list[16]
label = label_list[16]

seg = vm.segment_image(test_im, thresh = 40, preprocess = False)
skel, edges, bp = vm.skeletonize_vm(seg)
_, edge_labels = cv2.connectedComponents(edges)

segment_number = 37
diam, mean_diam, viz =  vm.visualize_vessel_diameter(edge_labels, segment_number, seg, test_im, use_label = False, pad = True)
viz[0,0] =5
vm.overlay_segmentation(test_im, viz, alpha = 1)
