#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:45:23 2020

Increases contrast of MIP images for annotation

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

data_path = '/home/sean/Documents/Calgary_postdoc/Data/czi_projections/Channel0/'
output_dir = '/home/sean/Documents/Calgary_postdoc/Data/czi_projections/Channel0/contrast_enhanced/'

file_list = glob.glob(data_path+'*.png')

for file in file_list:
    image = cv2.imread(file,0)
    img_norm = image/np.max(image)
    img_adj = np.floor(img_norm*255)
    file_separated = file.split('/')
    im_name = file_separated[-1]
    cv2.imwrite(output_dir + im_name,img_adj)
