#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:55:22 2021

region growing test

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from timeit import default_timer as timer
import sys

def on_mouse(event, x, y, flags, params):
    if event == cv.CV_EVENT_LBUTTONDOWN:
        print('Start Mouse Position: ' + str(x) + ', ' + str(y))
        s_box = x, y
        boxes.append(s_box)

def region_growing(img, starting_seed):
    #Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_threshold = 0.6
    region_size = 1
    intensity_difference = 0
    good_neighbors = 1

    seed = []
    seed.append(starting_seed)
    #Mean of the segmented region
    region_mean = img[starting_seed]

    #Input image parameters
    height, width = img.shape
    image_size = height * width

    #Initialize segmented output image
    segmented_img = np.zeros((height, width, 1), np.uint8)
    px_count = 0

    #Region growing until intensity difference becomes greater than certain threshold
    while (good_neighbors > 0):
        px_count +=1
        print(px_count)
        neighbor_points_list = []
        neighbor_intensity_list = []
        thresh_value = np.round(region_mean*region_threshold).astype(np.uint8)
        good_neighbors = 0
        #Loop through neighbor pixels
        for this_seed in seed:
            
            for i in range(4):
                #Compute the neighbor pixel position
                x_new = this_seed[0] + neighbors[i][0]
                y_new = this_seed[1] + neighbors[i][1]
    
                #Boundary Condition - check if the coordinates are inside the image
                check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)
    
                pixel_val = img[x_new, y_new]
                intensity_difference = np.abs(int(region_mean) - int(pixel_val))
                
                #Add neighbor if inside and not already in segmented_img
                if check_inside:
                    if segmented_img[x_new, y_new] == 0:
                        if intensity_difference <= thresh_value:
                            neighbor_points_list.append([x_new, y_new])
                            neighbor_intensity_list.append(img[x_new, y_new])
                            segmented_img[x_new, y_new] = 255
                            good_neighbors+=1
                            
            #New region mean
            region_pixels = np.argwhere(segmented_img == 255)
            region_mean = np.mean(img[region_pixels[:,0],region_pixels[:,1]])
                
            #Update the seed value
            seed = neighbor_points_list
    
    
    _, segmented_img = vm.fill_holes(segmented_img, 100)
    return segmented_img

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/sliding_window_annot/'
file = 'slice20'


image = cv2.imread(data_path + file + '/img.png',0)
label = cv2.imread(data_path + file + '/label.png',0)
label[label>0] = 1

img_size = np.shape(image)
img_small = cv2.resize(image,(np.int(img_size[1]/4),np.int(img_size[0]/4)))
seed = 158,611

output = region_growing(img_small, seed)

# test on 5 seeds

seeds = [[147, 512], [158, 611], [145, 727], [206, 579], [121, 613]] 

numel = []
for this_seed in seeds:
    this_seed = tuple(this_seed)
    output = region_growing(img_small, this_seed)
    numel.append(np.shape(np.argwhere(output>0))[0])
    
# test on consecutive slices
file_list = ['slice17','slice18', 'slice19','slice20']
seed = 158,611
output_img = []
numel = []
for file in file_list:
    this_label = cv2.imread(data_path + file + '/img.png',0)
    img_size = np.shape(image)
    resize_img = cv2.resize(this_label,(np.int(img_size[1]/4),np.int(img_size[0]/4)))
    temp_output = region_growing(resize_img,seed)
    output_img.append(temp_output)
    numel.append(np.shape(np.argwhere(temp_output>0))[0])

