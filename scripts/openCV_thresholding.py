#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:56:04 2020

https://www.youtube.com/watch?v=N81PCpADwKQ
OpenCV tutorial from code above

@author: sean
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


data_path ='/home/sean/Documents/Calgary_postdoc/Data/mCherryAug2020/'
img = cv2.imread(data_path + 'fish1_20ventral_kdrlhrasmcherry.tif',0)

_, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
_, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, th4 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)
_, th5 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO_INV)


cv2.imshow("Image",img)
cv2.imshow("Thresh",th1)
cv2.imshow("Thresh inv ",th2)
cv2.imshow("Thresh trunc",th3)
cv2.imshow("Thresh to zero",th4)
cv2.imshow("Thresh to zero inv",th5)


th6 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 2)
cv2.imshow("adaptive thresh mean c", th6)

th7 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 2)
cv2.imshow("adaptive thresh gauss c", th6)

cv2.waitKey(10000)
cv2.destroyAllWindows()

img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.xticks([]), plt.yticks([])
plt.imshow(th7)

titles = ['image', 'binary', 'binary inv', 'trunc', 'tozero', 'tozero inv']
images = [img, th1, th2, th3, th4, th5]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()


