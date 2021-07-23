#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:26:06 2021

plot_3d_vessel

@author: sean
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from czifile import CziFile
import numpy as np
import cv2

input_directory = '/home/sean/Documents/Calgary_postdoc/Data/raw_czi/'
input_file = 'flk gata inx 30hpf Feb 23 2019 E1 good qual.czi'

with CziFile(input_directory + input_file) as czi:
    image_arrays = czi.asarray()
