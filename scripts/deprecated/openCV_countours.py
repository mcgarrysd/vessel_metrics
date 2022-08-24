#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:35:45 2020

https://www.youtube.com/watch?v=N81PCpADwKQ
OpenCV tutorial from code above

@author: sean
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.close('all')
cv2.destroyAllWindows()

data_path ='/home/sean/Documents/Calgary_postdoc/Data/mCherryAug2020/'
img = cv2.imread(data_path + 'fish1_20ventral_kdrlhrasmcherry.tif',0)

ret, thresh = cv2.threshold(img, 20, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

print("number of countours " + str(len(contours)))
print(contours[0])

cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow('image',img)
cv2.imshow("threshold", thresh)


