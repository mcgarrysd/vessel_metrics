#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:31:26 2022

Skeleton figure

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
    
test_im = im_list[18]
label = label_list[18]

seg = vm.segment_image(test_im, thresh = 40, preprocess = False)

vm.overlay_segmentation(test_im, seg)
skel1 = skeletonize(seg)
skel2, _,_ = skeletonize_vm(seg)

skel_dif = skel2*2-skel1

kernel = np.ones((2,2),np.uint8)
skel1_d = cv2.dilate(skel1, kernel, iterations = 1)
skel2_d = cv2.dilate(skel2, kernel, iterations = 1)
skel_dif_d = cv2.dilate(skel_dif, kernel, iterations = 1)
