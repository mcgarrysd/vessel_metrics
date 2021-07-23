#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:39:43 2021

creates surface from image

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/sliding_window_annot/'
file = 'slice20'
image = cv2.imread(data_path + file + '/img.png',0)

img_size = np.shape(image)
img_small = cv2.resize(image,(np.int(img_size[1]/8),np.int(img_size[0]/8)))

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:img_small.shape[0], 0:img_small.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, img_small ,rstride=1, cstride=1, cmap=plt.cm.gray,
        linewidth=0)

# show it
plt.show()