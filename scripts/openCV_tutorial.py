#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:42:31 2020

https://www.youtube.com/watch?v=N81PCpADwKQ
OpenCV tutorial from code above

@author: sean
"""
import cv2
import numpy as np


data_path ='/home/sean/Documents/Calgary_postdoc/Data/mCherryAug2020/'
img = cv2.imread(data_path + 'fish1_20ventral_kdrlhrasmcherry.tif',1)
#flag loads color image, 1 for color, 0 for grayscale, -1 for unchanged

#check if image loaded properly
print(img)
print(img.shape)

#show image
cv2.imshow('Name',img)
cv2.waitKey(500) #show image for 5 seconds, 0 to wait for close
cv2.destroyAllWindows()
# cv2.imwrite(data_path + 'fish1_cherry_grayscale.png', img)

# k = cv2.waitKey(0)
#if k == 27:
#    cv2.destroyAllWindows()
    # 27 = escape key
#elif k == ord('s'):
#    cv2.imwrite(data_path + 'fish1_cherry_grayscale.png', img)

img_line = cv2.line(img, (0,0), (255,255), (255, 0 , 0), 10)
cv2.imshow('Name',img_line)
cv2.waitKey(50) #show image for 5 seconds, 0 to wait for close
cv2.destroyAllWindows()
# cv2.imwrite(data_path + 'fish1_cherry_line.png',img_line)


img_rectangle = cv2.rectangle(img,(255,255), (600,600), (0,0,255), 5)
cv2.imshow('Name',img_rectangle)
cv2.waitKey(5000) #show image for 5 seconds, 0 to wait for close
cv2.destroyAllWindows()
cv2.imwrite(data_path + 'fish1_cherry_rectangle.png',img_rectangle)

img = cv2.imread(data_path + 'fish1_20ventral_kdrlhrasmcherry.tif',1)
