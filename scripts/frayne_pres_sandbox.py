#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:41:07 2021

Frayne presentation sandbox

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
from skimage.morphology import skeletonize
from scipy.spatial import distance
from bresenham import bresenham
from copy import deepcopy

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/3dpf-9WT-4MM/'
sample_file = 'flk gata inx 3dpf Nov 14 2019 E5.czi'

czi = vm.preprocess_czi(data_path,sample_file)


def reslice_with_axis(volume, thickness, axis):
    return
