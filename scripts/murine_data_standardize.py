#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:56:35 2022

murine data standardize

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
import pandas as pd
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from scipy.stats import ttest_ind


data_path = '/media/sean/SP PHD U3/from_home/murine_data/raw/milene/'

out_path = '/media/sean/SP PHD U3/from_home/murine_data/milene/'

files = os.listdir(data_path)

for file in files:
    im = cv2.imread(data_path+file,0)
    im = vm.normalize_contrast(im)
    im = im.astype(np.uint8)
    cv2.imwrite(out_path+file, im)  
    
adam_path = '/media/sean/SP PHD U3/from_home/murine_data/raw/adam/'

out_path2 = '/media/sean/SP PHD U3/from_home/murine_data/adam/'

files = os.listdir(adam_path)

for file in files:
    im = cv2.imread(adam_path+file,0)
    im = vm.normalize_contrast(im)
    im = im.astype(np.uint8)
    cv2.imwrite(out_path2+file, im) 