#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:19:06 2022

runtime

@author: sean
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from aicsimageio import AICSImage
import timeit


t0 = timeit.default_timer()

data_path = '/media/sean/SP PHD U3/from_home/cynthia_network_length/oct_19/'

file_list = os.listdir(data_path+'Raw/')

t1 = timeit.default_timer()
elapsed_time = round(t1-t0,3)

print(f"Elapsed time: {elapsed_time}")

net_length = []
net_length_um = []
prefix_list = []
mkdir = True

fname = file.split('.')
fname = fname[0].split(' ')
und = '_'
prefix = fname[0]+und+fname[1]+und+fname[-2]+und+fname[-1]
prefix_list.append(prefix)
print(file, prefix)

volume, dims = vm.preprocess_czi(data_path+'Raw/',file, channel = 1)

t2 = timeit.default_timer()
elapsed_time = round(t2-t0,3)
print(f"Load file elapsed time: {elapsed_time}")

slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
this_slice = reslice[0]

t3 = timeit.default_timer()
elapsed_time = round(t3-t0,3)
print(f"Reslice elapsed time: {elapsed_time}")

seg = vm.brain_seg(this_slice, thresh = 40)

t4 = timeit.default_timer()
elapsed_time = round(t4-t0,3)
print(f"Seg elapsed time: {elapsed_time}")

skel, edges, bp = vm.skeletonize_vm(seg)
nl = vm.network_length(edges)
net_length.append(nl)
net_length_um.append(nl*dims[1])

t5 = timeit.default_timer()
elapsed_time = round(t5-t0,3)
print(f"Skeleton elapsed time: {elapsed_time}")

overlay = edges*100+seg*50

if mkdir == True:
    os.mkdir(data_path+'Processed/'+prefix)
cv2.imwrite(data_path+'Processed/'+prefix+'/img.png',this_slice)
cv2.imwrite(data_path+'Processed/'+prefix+'/label.png', overlay)