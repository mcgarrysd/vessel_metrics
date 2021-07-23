#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:18:19 2020

Kmeans segmentation test on zebrafish depth image

@author: sean
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.close('all')
cv2.destroyAllWindows()

data_path ='/home/sean/Documents/Calgary_postdoc/Data/mCherryAug2020/'
img = cv2.imread(data_path + 'fish1_20ventral_kdrlhrasmcherry.tif',0)

img_reshape = img.reshape((-1,1))
img_reshape = np.float32(img_reshape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
k = 5
attempts = 10

ret,label,center = cv2.kmeans(img_reshape,k,None, criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)

res = center[label.flatten()]
res = res.reshape(img.shape)

cv2.imshow('image',img)
cv2.imshow('segmentation',res)

label_reshape = label.reshape(img.shape)
plt.hist(label_reshape)
plt.show()