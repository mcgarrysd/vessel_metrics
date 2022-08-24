#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 08:10:00 2020

Generates samples from annotated mask

@author: sean
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage import data
from PIL import Image

data_path ='/home/sean/Documents/Calgary_postdoc/Data/czi_projections/Channel0/contrast_enhanced/Feb_23_E3/'
img = cv2.imread(data_path + 'img.png',0)
ground_truth = cv2.imread(data_path + 'label.png',0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
img = clahe.apply(img)

img = cv2.medianBlur(img, 11)

ground_truth_binary = ground_truth
ground_truth_binary[ground_truth>0] = 1
skeleton = skeletonize(ground_truth_binary)

plt.close('all')
plt.imshow(skeleton)


skel_int = skeleton.astype(np.uint8)
skel_int_display = skel_int * 255
cv2.imwrite(data_path + 'skeleton_image.png',skel_int_display)
skeleton_index = np.argwhere(skeleton==True)
 
inds = [3478, 5434, 7634]
for this_ind in inds:
    centroid = [skeleton_index[this_ind][0], skeleton_index[this_ind][1]]
    tile = np.empty(shape = (128,128))
    tile = img[centroid[0]-63:centroid[0]+64,centroid[1]-63:centroid[1]+64]
    tile = tile * 4
    tile_name = 'tile' + str(this_ind) + '.png'
    cv2.imwrite(data_path + tile_name, tile)
    
    tile_label = np.empty(shape = (128,128))
    tile_label = ground_truth[centroid[0]-63:centroid[0]+64,centroid[1]-63:centroid[1]+64]
    tile_label = tile_label*255
    tile_label_name = 'tile' + str(this_ind) + '_label.png'
    cv2.imwrite(data_path + tile_label_name, tile_label)


inds_consecutive = [1200, 1225, 1250]
for this_ind in inds_consecutive:
    centroid = [skeleton_index[0][this_ind], skeleton_index[1][this_ind]]
    tile = np.empty(shape = (128,128))
    tile = img[centroid[0]-63:centroid[0]+64,centroid[1]-63:centroid[1]+64]
    tile_name = 'tile' + str(this_ind) + '.png'
    cv2.imwrite(data_path + tile_name, tile)
    
    tile_label = np.empty(shape = (128,128))
    tile_label = ground_truth[centroid[0]-63:centroid[0]+64,centroid[1]-63:centroid[1]+64]
    tile_label_name = 'tile' + str(this_ind) + '_label.png'
    cv2.imwrite(data_path + tile_label_name, tile_label)

