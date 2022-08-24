#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:50:19 2020

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

img = cv2.pyrDown(img)

plt.hist(img)
plt.show()