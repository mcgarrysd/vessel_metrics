#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 08:43:08 2021

slic test

@author: sean
"""


import skimage.segmentation as seg
import cv2
import numpy as np
import vessel_metrics as vm
import matplotlib.pyplot as plt
import pandas as pd

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

im_name = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/hole_analysis/fish1_im1/img.png' 

img = cv2.imread(im_name,0)
segments = slic(img, n_segments = 500, sigma = 0, compactness = 0.001)

im_cl = vm.clahe(img)
seg_cl = slic(im_cl, n_segments = 300, sigma = 0, compactness = 0.001)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.imshow(mark_boundaries(im_cl,seg_cl))
plt.axis("off")

plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
im_enhanced = img*5
im_enhanced[im_enhanced>255] = 255
seg_enh = slic(im_cl, n_segments = 400, sigma = 0, compactness = 0.01)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.imshow(mark_boundaries(im_enhanced,seg_enh))
plt.axis("off")

plt.show()
