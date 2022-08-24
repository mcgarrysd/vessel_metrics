#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:49:19 2021

Analysis of plexus holes vs segmentation error

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

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/'
data_files = ['35M-59H inx 48hpf Apr 14 2019 E2.czi', '35M-59H inx 48hpf Apr 14 2019 E9.czi',\
'flk gata 48hpf Jul 26 2019 E5.czi', 'flk gata inx 48hpf Apr 14 2019 E4.czi']

vol = vm.preprocess_czi(data_path, data_files[0])
vol = vm.sliding_window(vol, 4)

im = vol[15,:,:]
im = im.astype(np.uint16)
raw_seg = vm.segment_vessels(im)

label= cv2.imread('/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/sample_images/boudegga/label.png',0)
label[label>0] = 1

hole_label = cv2.imread('/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/sample_images/boudegga/holes/label.png',0)
hole_label[hole_label<10] = 0
hole_label[hole_label>10] = 1

im_cl = vm.clahe(im)
seg = vm.segment_vessels(im_cl)

seg_hole_labels, hole_binary, stats = seg_holes(seg)
stats = stats[2:,:] #delete stats on background/vessels

hole_binary[seg_hole_labels == 1] = 0 # Deletes background
seg_hole_labels[seg_hole_labels == 1] = 0 

hole_overlay = hole_label + hole_binary*2

is_true = []
hole_mean = []
region_max = []
region_min = []
region_mean = []
eccentricity = []
area = []
perimeter = []

regions = regionprops(seg_hole_labels, im_cl)
label_index = []
for props in regions[2:]:
    area.append(props.area)
    hole_mean.append(props.mean_intensity)
    region_max.append(props.max_intensity)
    region_min.append(props.min_intensity)
    eccentricity.append(props.eccentricity)
    perimeter.append(props.perimeter)
    
    coords = props.coords
    vals = []
    label_vals = []
    for x,y in coords:
        vals.append(hole_label[x,y])
        label_vals.append(seg_hole_labels[x,y])
    is_true.append(mode(vals))
    label_index.append(mode(label_vals))
    
is_true = array(is_true)
true_inds = array(np.where(is_true == 1))
false_inds = np.where(is_true != 1)

area = array(area)
hole_mean = array(hole_mean)
region_max = array(region_max)
region_min = array(region_min)
eccentricity = array(eccentricity)
perimeter = array(perimeter)

area_true = area[is_true == 1]
area_false = area[is_true == 0]

hole_mean_t = hole_mean[is_true == 1]
hole_mean_f = hole_mean[is_true == 0]

region_max_t = region_max[is_true == 1]
region_max_f = region_max[is_true == 0]

region_min_t = region_min[is_true == 1]
region_min_f = region_min[is_true == 0]

eccentricity_t = eccentricity[is_true == 1]
eccentricity_f = eccentricity[is_true == 0]

perimeter_t = perimeter[is_true == 1]
perimeter_f = perimeter[is_true == 0]

fig, ax = plt.subplots(1,1)
_ = ax.hist(eccentricity_t);
_ = ax.hist(eccentricity_f)
plt.legend(('true','false'))
plt.title('eccentricity')

fig, ax = plt.subplots(1,1)
_ = ax.hist(area_true);
_ = ax.hist(area_false)
plt.legend(('true','false'))
plt.title('area')

fig, ax = plt.subplots(1,1)
_ = ax.hist(hole_mean_t, alpha = 0.5);
_ = ax.hist(hole_mean_f, alpha = 0.5)
plt.legend(('true','false'))
plt.title('mean_value')

fig, ax = plt.subplots(1,1)
_ = ax.hist(region_max_t);
_ = ax.hist(region_max_f)
plt.legend(('true','false'))
plt.title('region max')

fig, ax = plt.subplots(1,1)
_ = ax.hist(region_min_t);
_ = ax.hist(region_min_f)
plt.legend(('true','false'))
plt.title('region min')

fig, ax = plt.subplots(1,1)
_ = ax.hist(perimeter_t);
_ = ax.hist(perimeter_f)
plt.legend(('true','false'))
plt.title('perimeter')

surface_area = area/perimeter
sa_t = surface_area[is_true == 1]
sa_f = surface_area[is_true == 0]

fig, ax = plt.subplots(1,1)
_ = ax.hist(sa_t, alpha = 0.5);
_ = ax.hist(sa_f, alpha = 0.5)
plt.legend(('true','false'))
plt.title('surface_area')

def seg_holes(label):
    label_inv = np.zeros_like(label)
    label_inv[label == 0] = 1
    label_inv = label_inv.astype(np.uint8)
    _, inverted_labels, stats, _ = cv2.connectedComponentsWithStats(label_inv)
    return inverted_labels, label_inv, stats

these_inds = np.where(surface_area < 15) and np.where(area < 6000)
true_stratification = is_true[these_inds]

for y in 

low_sa = np.where(surface_area <15)
low_area = np.where(area < 6000)


filled_holes = np.zeros_like(label)
for x,y in zip(hole_mean, label_index):
    if x>3:
        filled_holes[seg_hole_labels == y] = 1

new_seg_disp = deepcopy(seg)
new_seg_disp = new_seg_disp+filled_holes*2
plt.figure(); plt.imshow(new_seg_disp)

new_seg = deepcopy(seg)
new_seg = new_seg+filled_holes
new_jacc = vm.jaccard(label,new_seg)

fig, ax = plt.subplots(1,1)
ax.scatter(sa_t, hole_mean_t, label = 'true')
ax.scatter(sa_f, hole_mean_f, label = 'false')
plt.xlabel('surface area'); plt.ylabel('mean value')

segmented_volume = np.zeros_like(vol)
for z in range(shape(vol)[0]):
    segmented_volume[z,:,:] = vm.segment_vessels(vol[z,:,:])
    
def sliding_window_label(volume,thickness):
    num_slices = np.shape(volume)[0]
    out_slices = num_slices-thickness
    output = np.zeros([out_slices, np.shape(volume)[1], np.shape(volume)[2]])
    for i in range(0,out_slices):
        im_chunk = volume[i:i+thickness,:,:]
        output[i,:,:] = np.sum(im_chunk,axis = 0)
    return output

sw_label = sliding_window_label(segmented_volume,4)
sw_slice = sw_label[12,:,:]

sw_consolidated = np.zeros_like(sw_slice)
sw_consolidated[sw_slice>2] = 1

plt.figure(); plt.imshow(sw_consolidated)
sw_jacc = vm.jaccard(label,sw_consolidated)
