#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:32:29 2021

Corneal image test

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from timeit import default_timer as timer
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from scipy import stats
from scipy.spatial import distance
from skimage.draw import line
from copy import deepcopy

data_path = '/home/sean/Documents/vessel_metrics/data/cornea/'

im_names = ['Img0226.jpg', 'Img0230.jpg']

im_list = []
for i in im_names:
    im_list.append(cv2.imread(data_path+i))

test_im = deepcopy(im_list[0])
plt.imshow(test_im)

plt.imshow(test_im[:,:,2])

test_im[:,:,2] = 0

def rgb2gray(im):
    output = np.zeros_like(im[:,:,0])
    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            output[x,y] = im[x,y,0]*0.2989+im[x,y,1]*0.587+im[x,y,2]*0.1140
    return output

test_gray1 = rgb2gray(test_im)
test_gray2 = rgb2gray(im_list[0])

plt.figure(); plt.imshow(test_gray1)
plt.figure(); plt.imshow(test_gray2) 