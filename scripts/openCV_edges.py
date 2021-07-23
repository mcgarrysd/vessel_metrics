#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 08:19:06 2020

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

lap = cv2.Laplacian(img, cv2.CV_64F, ksize = 3) # 64 bit float
lap = np.uint8(np.absolute(lap))

sobelx = cv2.Sobel(img, cv2.CV_64F, dx = 1, dy =  0, ksize = 1)
sobelx = np.uint8(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, dx = 0 , dy = 1, ksize = 1)
sobely = np.uint8(sobely)

sobel_both = cv2.bitwise_or(sobelx, sobely)

titles = ['image', 'Laplacian', 'sobel x', 'sobel y', 'sobel combined']
images = [img, lap, sobelx, sobely, sobel_both]

for i in range(5):
    plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()

plt.figure(2)

canny = cv2.Canny(img,50, 225)

titles_edge = ['canny']
images_edge = [canny]

for i in range(1):
    plt.subplot(1,1,i+1), plt.imshow(images_edge[i], 'gray')
    plt.title(titles_edge[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()



