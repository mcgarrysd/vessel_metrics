#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:36:01 2022

idy seg test

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

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/idy_annotate/'

im = cv2.imread(data_path+'img.png',0)
seg = vm.brain_seg(im, filter = 'meijering', thresh = 40)
idy_label = cv2.imread(data_path+'idy_label2.tiff',0)
label = cv2.imread(data_path+'label.png',0)

idy_label2 = idy_label[:,342:2390]
idy_label2 = cv2.resize(idy_label2, [1024,1024])

idy_label3 = np.zeros_like(idy_label2)
idy_label3[idy_label2<200]=1


idy_label_fixed = np.zeros_like(idy_label)
idy_label_fixed[idy_label<200] = 1