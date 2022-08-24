#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:23:39 2020

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

lr = cv2.pyrDown(img)

cv2.imshow("pyramid 1",lr)

lr2 = cv2.pyrDown(lr)

cv2.imshow("pyramid 2", lr2)

hr = cv2.pyrUp(lr2)
cv2.imshow("higher res 1",hr)

layer = img.copy()
gauss_pyramid = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gauss_pyramid.append(layer)
    cv2.imshow(str(i), layer)


