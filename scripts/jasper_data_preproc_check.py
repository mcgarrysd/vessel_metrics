#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 07:11:35 2021

jasper data preprocess check

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
from copy import deepcopy
import os
from shutil import copyfile

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/'
out_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/czis/'

test_im = 'czis/3dpf_fish5_mutant.czi'

vol = vm.preprocess_czi(data_path, test_im)
vol = vm.sliding_window(vol,4)

plt.imshow(vol[11,:,:])
test_seg = vm.segment_vessels(vol[11,:,:], bin_thresh = 10)
