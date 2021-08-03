#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:53:48 2021

Hole analysis - larger dataset

@author: sean
"""

import cv2
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
import matplotlib.pyplot as plt
from statistics import mode
import pandas as pd
from skimage.measure import regionprops
from skimage.measure import regionprops_table
from copy import deepcopy
from sklearn import svm

import time
start_time = time.time()

cv2.setRNGSeed(1)

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/hole_analysis/'
data_files = ['35M-59H inx 48hpf Apr 14 2019 E2.czi', '35M-59H inx 48hpf Apr 14 2019 E9.czi',\
'flk gata 48hpf Jul 26 2019 E5.czi', 'flk gata inx 48hpf Apr 14 2019 E4.czi']

outpath_flag = 0
if outpath_flag:
    outpath = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/hole_analysis/'
    vol = vm.preprocess_czi(data_path, data_files[0])
    vol = vm.sliding_window(vol, 4)
    vol = vol.astype(np.uint8)
    im1 = vol[15,:,:]
    im2 = vol[10,:,:]
    
    cv2.imwrite(outpath + 'fish1_im1.png',im1)
    cv2.imwrite(outpath + 'fish1_im2.png',im2)
    
    vol = vm.preprocess_czi(data_path, data_files[1])
    vol = vm.sliding_window(vol, 4)
    
    im3 = vol[15,:,:]
    im4 = vol[20,:,:]
    
    cv2.imwrite(outpath + 'fish2_im1.png',im3)
    cv2.imwrite(outpath + 'fish2_im2.png',im4)
    
    vol = vm.preprocess_czi(data_path, data_files[2])
    vol = vm.sliding_window(vol, 4)
    vol = vol.astype(np.uint8)
    im5 = vol[15,:,:]
    im6 = vol[10,:,:]
    
    cv2.imwrite(outpath + 'fish3_im1.png',im5)
    cv2.imwrite(outpath + 'fish3_im2.png',im6)
    
dir_list = ['fish1_im1/', 'fish1_im2/', 'fish3_im1/', 'fish3_im2/']
im_list = []; label_list = []

is_true = []
hole_mean = []
region_max = []
region_min = []
region_mean = []
eccentricity = []
area = []
perimeter = []
fish_list = []
seg_list = []
inv_label_list = []
overlap_list = []
label_index = []

for i in dir_list:
    im = cv2.imread(data_path + i + 'img.png',0)
    label = cv2.imread(data_path + i + 'label.png',0)
    
    label[label>1] = 1
    im = vm.clahe(im)
    
    seg = vm.segment_vessels(im)
    
    labelled_holes, inv_label, stats = vm.seg_holes(seg)
    for s in range(len(stats)):
        if stats[s,4]>50000:
            inv_label[labelled_holes == s] = 0
            labelled_holes[labelled_holes == s] =0
    overlap = label+inv_label*2
    im_list.append(im)
    label_list.append(label)
    seg_list.append(seg)
    inv_label_list.append(inv_label)
    overlap_list.append(overlap)
    
    regions = regionprops(labelled_holes, im)
    for props in regions[2:]:
        area.append(props.area)
        hole_mean.append(props.mean_intensity)
        region_max.append(props.max_intensity)
        region_min.append(props.min_intensity)
        eccentricity.append(props.eccentricity)
        perimeter.append(props.perimeter)
        fish_list.append(i)
        
        coords = props.coords
        vals = []
        label_vals = []
        for x,y in coords:
            vals.append(label[x,y])
            label_vals.append(labelled_holes[x,y])
        is_true.append(mode(vals))
        label_index.append(mode(label_vals))

surface_area = np.array(area)/np.array(perimeter)

col_names = ['area','mean','max','min','eccentricity','perimeter','surface area', 'fish_list','is_true','index']
df = pd.DataFrame(list(zip(area,hole_mean,region_max,region_min,eccentricity,perimeter,surface_area, fish_list,is_true,label_index)),columns = col_names)

clf = svm.SVC()

for f in fish_list:
    train = df[df['fish_list']!=f]
    test = df[df['fish_list']==f]
    data_cols = col_names[0:7]
    train_data = train[data_cols]
    test_data = test[data_cols]
    
    train_label = train['is_true']
    test_label = test['is_true']
    
    clf.fit(train_data,train_label)
    predicted_labels = clf.predict(test_data)
    
    data_cols_reduced = ['area','mean','surface area']
    train_reduced = train[data_cols_reduced]
    test_reduced = test[data_cols_reduced]
    
    clf.fit(train_reduced, train_label)
    predicted_reduced = clf.predict(test_reduced)

print(time.time() - start_time)

# test test test
