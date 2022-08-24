#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 21:54:03 2022

Emma data test

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

data_path = '/media/sean/SP PHD U3/from_home/emma_data/raw_data/' 
groups = os.listdir(data_path)


all_files = []
ovl_perc = []
vessel_density = []
im_list = []
seg_list = []
raw_im_list = []
for g in groups:
    data_files = os.listdir(data_path+g+'/')
    for file in data_files:
        volume = vm.preprocess_czi(data_path+g+'/',file, channel = 1)
        slice_range = len(volume)
        reslice = vm.reslice_image(volume,slice_range)
        this_slice = reslice[0]
        raw_im_list.append(this_slice)
        this_slice = vm.preprocess_seg(this_slice.astype(np.uint8))
        s = range(15,30,5)
        seg = vm.brain_seg(this_slice, sigmas = s, filter = 'jerman', thresh = 30, preprocess = False, ditzle_size = 1000)
        enh = vm.seg_no_thresh(this_slice, sigmas = s, filter = 'jerman', preprocess = False)
        #vm.overlay_segmentation(this_slice, seg)
        
        all_files.append(file)
        im_list.append(this_slice)
        seg_list.append(seg)
        
        
        v2 = vm.preprocess_czi(data_path+g+'/',file, channel = 0)
        reslice_ch0 = vm.reslice_image(v2,slice_range)
        that_slice = reslice_ch0[0]
        that_slice = vm.preprocess_seg(that_slice.astype(np.uint8))
        enh2 = vm.seg_no_thresh(that_slice, sigmas = s, filter = 'jerman', preprocess = False)
        seg_ch0 = np.zeros_like(enh2)
        seg_ch0[enh2>100]=1
        
        ovl_perc.append(channel_overlap(seg, seg_ch0))
        
#########################################################################
# diameters
im = raw_im_list[0]
seg = seg_list[0]

skel, edges, bp = vm.skeletonize_vm(seg)
edge_count, edge_labels = cv2.connectedComponents(edges)
diams, mean_diam, viz = vm.visualize_vessel_diameter(edge_labels, 5, seg, im, use_label = False)

cv2.imwrite('/media/sean/SP PHD U3/from_home/emma_data/test_im.png',im)

vm.overlay_segmentation(im, skel)

cropped_ims = []
cropped_seg = []
cropped_edges = []
for im, seg in zip(im_list, seg_list):
    skel, edges, bp = vm.skeletonize_vm(seg)
    roi = vm.generate_roi(im)
    cropped_ims.append(vm.crop_roi(im, roi))
    cropped_seg.append(vm.crop_roi(seg, roi))
    cropped_edges.append(vm.crop_roi(edges, roi))

#####################################################################
# save ims
    
output_path = '/media/sean/SP PHD U3/from_home/emma_data/processed/'
mkdir = True
for i in range(len(im_list)):
    file = all_files[i]
    prefix = file.split('.')[0]
    prefix = prefix.replace(" ","_")
    prefix = prefix.replace("#","")
    prefix = prefix.replace("-","")
    print(prefix)
    
    if mkdir == True:
        os.mkdir(output_path+prefix)
    cv2.imwrite(output_path+prefix+'/im_raw.png',raw_im_list[i])
    cv2.imwrite(output_path+prefix+'/im.png',im_list[i])
    cv2.imwrite(output_path+prefix+'/full_seg.png',seg_list[i])
    cv2.imwrite(output_path+prefix+'/cropped_im.png',cropped_ims[i])
    cv2.imwrite(output_path+prefix+'/cropped_seg.png',cropped_seg[i])
    
    
###################################################################
# Diameter analysis on cropped images

data_path = '/media/sean/SP PHD U3/from_home/emma_data/processed/GFP_–_Apr_29_22_1/'

full_im = cv2.imread(data_path+'im.png',0)
im = cv2.imread(data_path+'cropped_im.png',0)
seg = cv2.imread(data_path+'cropped_seg.png',0)

pad_size = 25
seg = np.pad(seg, pad_size)
im = np.pad(im,pad_size)

skel, edges, bp = vm.skeletonize_vm(seg)

vm.overlay_segmentation(im,edges)

_, edge_labels = cv2.connectedComponents(edges)

values, counts = np.unique(edge_labels, return_counts = True)
values = values[1:]
counts = counts[1:]
longest_seg = values[counts == np.max(counts)][0]
diams, mean_diam, viz = vm.visualize_vessel_diameter(edge_labels, longest_seg, seg, im)
vm.overlay_segmentation(im, viz)








###########################################################
for im, edges, seg in zip (cropped_ims, cropped_edges, cropped_seg):
    _, edge_labels = cv2.connectedComponents(edges)
    values, counts = np.unique(edge_labels, return_counts = True)
    values = values[1:]
    counts = counts[1:]
    longest_seg = values[counts == np.max(counts)][0]
    diams, mean_diam, viz = vm.visualize_vessel_diameter(edge_labels, longest_seg, seg, im, use_label = False, pad = True)
    vm.overlay_segmentation(im, viz)
    
def channel_overlap(ch0, ch1):
    num_ch0 = len(np.argwhere(ch0 == 1))
    overlap = ch0+ch1
    num_ovl = len(np.argwhere(overlap == 2))
    ovl_perc = num_ovl/num_ch0
    return ovl_perc




def visualize_vessel_diameter(edge_labels, segment_number, seg, im, use_label = False, pad = True):
    if pad == True:
        pad_size = 25
        edge_labels = np.pad(edge_labels,pad_size)
        seg = np.pad(seg, pad_size)
        im = np.pad(im,pad_size)
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==segment_number] = 1
    segment_median = vm.segment_midpoint(segment)

    vx,vy = vm.tangent_slope(segment, segment_median)
    bx,by = vm.crossline_slope(vx,vy)
    
    viz = np.zeros_like(seg)
    cross_length = vm.find_crossline_length(bx,by, segment_median, seg)
    
    if cross_length == 0:
        diameter = 0
        mean_diameter = 0
        return diameter, mean_diameter, viz
    
    diameter = []
    segment_inds = np.argwhere(segment)
    for i in range(10,len(segment_inds),10):
        this_point = segment_inds[i]
        vx,vy = tangent_slope(segment, this_point)
        bx,by = crossline_slope(vx,vy)
        _, cross_index = make_crossline(bx,by, this_point, cross_length)
        if use_label:
            cross_vals = crossline_intensity(cross_index,seg)
            diam = label_diameter(cross_vals)
        else:
            cross_vals = crossline_intensity(cross_index, im)
            diam = fwhm_diameter(cross_vals)
        if diam == 0:
            val = 5
        else:
            val = 10
        for ind in cross_index:
            viz[ind[0], ind[1]] = val
        diameter.append(diam)
    diameter = [x for x in diameter if x != 0]
    if diameter:
        mean_diameter = np.mean(diameter)
    else:
        mean_diameter = 0
    
    if pad == True:
        im_shape = edge_labels.shape
        viz = viz[pad_size:im_shape[0]-pad_size,pad_size:im_shape[1]-pad_size]
    return diameter, mean_diameter, viz


def find_crossline_length(bx,by,point,seg):
    distance = 5
    diam = 0
    im_size = seg.shape[0]
    while diam == 0:
        distance +=5
        coords, cross_index = vm.make_crossline(bx,by,point,distance)
        out = vm.plot_crossline(seg, cross_index, bright = True)
        if all(i<im_size for i in coords):
            print(distance)
            seg_val = []
            for i in cross_index:
                seg_val.append(seg[i[0], i[1]])
            steps = np.where(np.roll(seg_val,1)!=seg_val)[0]
            if steps.size>0:
                if steps[0] == 0:
                    steps = steps[1:]
                num_steps = len(steps)
                if num_steps == 2:
                    diam = abs(steps[1]-steps[0])
            if distance >100:
                break
        else:
            break
    length = diam*2.5
    return length