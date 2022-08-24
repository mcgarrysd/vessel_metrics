#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:56:19 2022

FA_data_v2

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
from bresenham import bresenham
import itertools

data_path = '/media/sean/SP PHD U3/from_home/RECOVERY-FA19/' 

im_name = 'Img01_RECOVERY-FA19.tif'
label_name = 'Label01_RECOVERY-FA19.png'


im = cv2.imread(data_path+'images/'+im_name,0)
label = cv2.imread(data_path+'labels/'+label_name,0)
label[label>0] =1