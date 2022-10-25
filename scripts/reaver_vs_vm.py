#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:44:46 2022

reaver vs  vm

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

data_path = '/home/sean/Documents/from_home/vm_manuscript/reaver_test_data/all_ims_preproc/'

data_files = os.listdir(data_path)
im_files = [i for i in data_files if '.png' in i]

seg_files = os.listdir(data_path+'bw/')

manual_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/E1_segmentation/' 

im_files = sorted(im_files)
seg_files = sorted(seg_files)

im_list = []
seg_list = []
jacc_list = []
for i,j in zip(im_files, seg_files):
    im = cv2.imread(data_path+i,0)
    base_name = i.split('.')[0]
    reaver_seg = cv2.imread(data_path+'bw/'+j,0)
    reaver_seg[reaver_seg>0]=1
    
    manual_seg = cv2.imread(manual_path+base_name+'/label.png',0)
    
    manual_seg[manual_seg>0]=1
    
    vm_seg = vm.brain_seg(im, thresh = 40)
    
    jacc_list.append(vm.jaccard(manual_seg, reaver_seg))