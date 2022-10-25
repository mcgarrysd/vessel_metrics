#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:50:48 2022

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
from scipy.stats import ttest_ind

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/reaver_data/'
data_files = os.listdir(data_path)

manual_list = []
im_list = []
reaver_list = []
vm_list = []
aq_list = []
at_list = []
rave_list = []

j_at = []
j_aq = []
j_rave = []
j_reaver = []
j_vm = []
for file in data_files:
    im = cv2.imread(data_path+file+'/orig.tif',0)
    manual = cv2.imread(data_path+file+'/manual.tif',0)
    at = cv2.imread(data_path+file+'/AngioTool.tif',0)
    aq = cv2.imread(data_path+file+'/angioquant.tif',0)
    rave = cv2.imread(data_path+file+'/RAVE.tif',0)
    reaver = cv2.imread(data_path+file+'/reaver.tif',0)
    
    manual[manual<50]=0
    manual[manual>50]=1
    at[at<50]=0
    at[at>50]=1
    aq[aq<50]=0
    aq[aq>50]=1
    rave[rave<50]=0
    rave[rave>50]=1
    reaver[reaver<50]=0
    reaver[reaver>50]=1
    
    im_list.append(im)
    at_list.append(at)
    aq_list.append(aq)
    rave_list.append(rave)
    reaver_list.append(reaver)
    
    seg = vm.multi_scale_seg(im, thresh = 60, ditzle_size = 1000)
    
    j_at.append(vm.jaccard(manual, at))
    j_aq.append(vm.jaccard(manual, aq))
    j_rave.append(vm.jaccard(manual, rave))
    j_reaver.append(vm.jaccard(manual, reaver))
    j_vm.append(vm.jaccard(manual, seg))
    
