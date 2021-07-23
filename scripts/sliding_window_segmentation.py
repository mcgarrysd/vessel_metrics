#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 08:20:27 2021

sliding_window_segmentation - tests segmentation accuracy on 
consecutive slices, resliced to MIP of 4 standard slices

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from timeit import default_timer as timer

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/sliding_window_annot/'

os.chdir(data_path)
file_list = []
for file in glob.glob("slice*"):
    file_list.append(file)

jacc = []
start = timer()
for file in file_list:

    image = cv2.imread(data_path + file + '/img.png',0)
    label = cv2.imread(data_path + file + '/label.png',0)
    label[label>0] = 1
    
    test_label = segment_vessels(image, k = 12, hole_size = 500, ditzle_size = 750)
    overlay = test_label*2 + label
    
    jacc.append(vm.jaccard(label,test_label))
    
    plt.figure()
    plt.imshow(test_label)
    plt.figure()
    plt.imshow(overlay)
    plt.figure()
    plt.imshow(image)

end = timer()
print(end-start)
jacc = np.asarray(jacc)
mean_jaccard = np.mean(jacc)

