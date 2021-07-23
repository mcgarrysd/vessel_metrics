#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:34:21 2021

tests segmentation on retinal vessels

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
from cv2_rolling_ball import subtract_background_rolling_ball

data_path = '/home/sean/Documents/Calgary_postdoc/Data/'
file = 'kdrl 5x5 eye 2 good.czi'

with CziFile(data_path + file) as czi:
    image_arrays = czi.asarray()

image = np.squeeze(image_arrays)
image = vm.normalize_contrast(image)

img_size = np.shape(image)
img_small = cv2.resize(image,(np.int(img_size[1]/8),np.int(img_size[0]/8)))

img_ball, background = subtract_background_rolling_ball(img_small, 400, light_background=False, use_paraboloid=False, do_presmooth=True)


test_label = vm.segment_vessels(img_small)
