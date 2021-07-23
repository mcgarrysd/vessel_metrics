#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:48:09 2020

https://www.youtube.com/watch?v=N81PCpADwKQ
OpenCV tutorial from code above

@author: sean
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

data_path ='/home/sean/Documents/Calgary_postdoc/Data/mCherryAug2020/'
img = cv2.imread(data_path + 'fish1_20ventral_kdrlhrasmcherry.tif',0)

_, mask = cv2.threshold(img,20, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)

dilation = cv2.dilate(mask, kernel, iterations = 1)
erosion = cv2.erode(mask, kernel, iterations = 1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)

titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'gradient', 'tophat']
images = [img, mask, dilation, erosion, opening, closing, gradient, tophat]

for i in range(8):
    plt.subplot(2,4, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()

