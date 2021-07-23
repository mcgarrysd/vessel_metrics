#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 08:55:56 2020

test script to begin working with czi files

@author: sean
"""

from czifile import CziFile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from PIL import Image

with CziFile('/home/sean/Documents/Calgary_postdoc/Data/mCherryAug2020/Fish1_20ventral_9dpf.czi') as czi:
    image_arrays = czi.asarray()
    
print(image_arrays.shape)
print(type(image_arrays))

image = np.squeeze(image_arrays)
print(image.shape)

im_slice = image[35,:,:]
print(im_slice.shape)

f = plt.figure(1)
plt.imshow(im_slice)
f.show()

g = plt.figure(2)
plt.hist(im_slice)
g.show()

h = plt.figure(3)
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
cl1 = clahe.apply(im_slice)
plt.imshow(cl1)
h.show()

i = plt.figure(4)
clahe2 = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (50,50))
cl2 = clahe2.apply(im_slice)
plt.imshow(cl2)
i.show()

j = plt.figure(5)
clahe3 = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (100,100))
cl3 = clahe3.apply(im_slice)
plt.imshow(cl3)
j.show()

im1 = Image.fromarray(im_slice)
im1.save('/home/sean/Documents/Calgary_postdoc/Data/mCherryAug2020/slice35.png')
im2 = Image.fromarray(cl2)
im2.save('/home/sean/Documents/Calgary_postdoc/Data/mCherryAug2020/slice35_clahe.png')

