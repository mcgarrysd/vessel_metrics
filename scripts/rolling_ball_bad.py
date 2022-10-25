#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:40:17 2022

rolling ball is bad

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
from skimage.morphology import white_tophat, black_tophat, disk
import timeit
from skimage import data, restoration, util

data_path = '/media/sean/SP PHD U3/from_home/cynthia_network_length/oct_19/'

file_list = os.listdir(data_path+'Raw/')
file = file_list[0]

volume, dims = vm.preprocess_czi(data_path+'Raw/',file, channel = 1)
slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
this_slice = reslice[0]

t0 = timeit.default_timer()
old_preproc = preprocess_seg(this_slice)
timer_output(t0)

t0 = timeit.default_timer()
new_preproc = preprocess_subtract(this_slice)
timer_output(t0)

bg = subtract_background(this_slice, radius = 50, light_bg = False)


test = vm.normalize_contrast(this_slice)
bg = restoration.rolling_ball(test, radius = np.round(test.shape[0]/3))
rolling_im = test-bg


def preprocess_seg(image,ball_size = 0, median_size = 7, upper_lim = 255, lower_lim = 0, bright_background = False):
    image = vm.normalize_contrast(image)
    if ball_size == 0:
        ball_size = np.round(image.shape[0]/3)
    
    if bright_background == False:
        bg = restoration.rolling_ball(image, radius = ball_size)
        image = image-bg
    else:
        image_inverted = util.invert(image)
        bg_inv = restoration.rolling_ball(image, radius = ball_size)
        image = util.invert(image_inverted-bg_inv)
    image = cv2.medianBlur(image.astype(np.uint8),median_size)
    image = vm.contrast_stretch(image, upper_lim = upper_lim, lower_lim = lower_lim)
    return image

def preprocess_subtract(image,radius = 50, median_size = 7, upper_lim = 255, lower_lim = 0, bright_background = False):
    image = vm.normalize_contrast(image)

    image = subtract_background(image, radius = radius, light_bg = bright_background)
    image = cv2.medianBlur(image.astype(np.uint8),median_size)
    image = vm.contrast_stretch(image, upper_lim = upper_lim, lower_lim = lower_lim)
    return image

def subtract_background(image, radius=50, light_bg=False):
    str_el = disk(radius) 
    if light_bg:
        output =  black_tophat(image, str_el)
    else:
        output = white_tophat(image, str_el)
    return output
        
def timer_output(t0):
    t1 = timeit.default_timer()
    elapsed_time = round(t1-t0,3)
    
    print(f"Elapsed time: {elapsed_time}")