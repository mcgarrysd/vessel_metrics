#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 07:58:30 2020

https://www.youtube.com/watch?v=N81PCpADwKQ
OpenCV tutorial from code above

@author: sean
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.close('all')

data_path ='/home/sean/Documents/Calgary_postdoc/Data/mCherryAug2020/'
img = cv2.imread(data_path + 'fish1_20ventral_kdrlhrasmcherry.tif',0)

x = 5; y = 5
kernel = np.ones((x,y), np.float32)/(x*y)

dst = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (x,y))
gblur = cv2.GaussianBlur(img, (x,y), 0)
median = cv2.medianBlur(img, x) # kernel must be an odd number
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

titles = ['images', '2D convolution', 'blur', 'Gauss', 'Median', 'Bilateral']
images = [img, dst, blur, gblur, median, bilateral]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()

