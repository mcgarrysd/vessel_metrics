#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 09:20:41 2021

kmeans comparison

@author: sean
"""

import cv2
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
import matplotlib.pyplot as plt
from statistics import mode
import pandas as pd
from skimage.measure import regionprops
from skimage.measure import regionprops_table
from copy import deepcopy
from sklearn import svm
from sklearn.cluster import KMeans
from cv2_rolling_ball import subtract_background_rolling_ball

data_path = '/home/sean/Documents/hole_analysis/'

dir_list = ['fish1_im1/', 'fish1_im2/', 'fish3_im1/', 'fish3_im2/']

for i in dir_list:
    im = cv2.imread(data_path + i + 'img.png',0)
    im = vm.clahe(im)
    
    image = cv2.medianBlur(im.astype(np.uint8),7)
    image, background = subtract_background_rolling_ball(image, 400, light_background=False, use_paraboloid=False, do_presmooth=True)
    im_vector = image.reshape((-1,)).astype(np.float32)
    k = 12
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label_pp, center = cv2.kmeans(im_vector,k,None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    _, label_ran, center = cv2.kmeans(im_vector,k,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    km_skl = KMeans(n_clusters = k, init = 'k-means++').fit(im_vector)
    label_skl = km_skl.predict(im_vector)
