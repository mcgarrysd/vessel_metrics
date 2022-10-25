#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:35:15 2022

vessel diameter error tolerance

@author: sean
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import distance
from skimage import draw
import matplotlib.pyplot as plt
import os
from czifile import CziFile
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.draw import line
from bresenham import bresenham
from skimage.util import invert
from skimage.filters.ridges import compute_hessian_eigenvalues
import vessel_metrics as vm
from scipy.stats import kurtosis


im = cv2.imread('/media/sean/SP PHD U3/from_home/murine_data/adam/027.tif' ,0)
im_preprocess = vm.preprocess_seg(im)
sigma1 = range(1,5,1); sigma2 = range(10,20,5)
seg = vm.multi_scale_seg(im, sigma1 = sigma1, sigma2 = sigma2, filter = 'meijering', thresh = 40, ditzle_size = 0, hole_size = 200)

skel, edges, bp = vm.skeletonize_vm(seg)
_, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))

vm.overlay_segmentation(im, edge_labels)

test_seg_easy = [69, 46, 32]
test_seg_hard = [51, 22, 24]

easy_seg1_mean = np.mean([21.95, 26.32, 25.18])
easy_seg2_mean = np.mean([18.25, 18.71, 18.03])
easy_seg3_mean = np.mean([24.98, 25., 27.28])

diams_easy = []
mean_diams_easy = []
viz_easy = []
cross_vals_easy = []
for this_seg in test_seg_easy:
    temp_diam, temp_mean_diam, temp_viz, temp_cv = visualize_vessel_diameter_debug(edge_labels,this_seg, seg, im)
    diams_easy.append(temp_diam)
    mean_diams_easy.append(temp_mean_diam)
    viz_easy.append(temp_viz)
    cross_vals_easy.append(temp_cv)
    
    
diams_easy2 = []
mean_diams_easy2 = []
viz_easy2 = []
cross_vals_easy2 = []
for this_seg in test_seg_easy:
    temp_diam, temp_mean_diam, temp_viz, temp_cv = visualize_vessel_diameter_debug(edge_labels,this_seg, seg, im_preprocess)
    diams_easy2.append(temp_diam)
    mean_diams_easy2.append(temp_mean_diam)
    viz_easy2.append(temp_viz)
    cross_vals_easy2.append(temp_cv)



diams_hard = []
mean_diams_hard = []
viz_hard = []
cross_vals_hard = []
for this_seg in test_seg_hard:
    temp_diam, temp_mean_diam, temp_viz, temp_cv = visualize_vessel_diameter(edge_labels,this_seg, seg, im)
    diams_hard.append(temp_diam)
    mean_diams_hard.append(temp_mean_diam)
    viz_hard.append(temp_viz)
    cross_vals_easy.append(temp_cv)


seg1_cv = cross_vals_easy[0]
seg1_diams = diams_easy[0]

output_seg1, outlier_seg1 = detect_diameter_outliers(diams_easy2[0])

diams_easy3 = []
mean_diams_easy3 = []
viz_easy3 = []
cross_vals_easy3 = []
for this_seg in test_seg_easy:
    temp_diam, temp_mean_diam, temp_viz, temp_cv = visualize_vessel_diameter_debug(edge_labels,this_seg, seg, im)
    diams_easy3.append(temp_diam)
    mean_diams_easy3.append(temp_mean_diam)
    viz_easy3.append(temp_viz)
    cross_vals_easy3.append(temp_cv)
###########################################################


def detect_diameter_outliers(diam_list, tolerance = 0.5):
    mean_diameter = np.mean(diam_list)
    min_diam = mean_diameter-(mean_diameter*tolerance)
    max_diam = mean_diameter+(mean_diameter*tolerance)
    output = []
    is_outlier = []
    for d in diam_list:
        if d>min_diam and d<max_diam:
            output.append(d)
            outlier = 0
        else:
            outlier = 1
        is_outlier.append(outlier)
    return output, is_outlier


def remove_outlier_crosslines(edge_labels, segment_number, seg, im, is_outlier, cross_length, use_label = False):
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==segment_number] = 1
    segment_inds = np.argwhere(segment)
    viz = np.zeros_like(seg)
    for i,j in zip(list(range(10,len(segment_inds),10)), is_outlier):
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
            all_cross_vals.append(cross_vals)
        if diam == 0 or j == 1:
            val = 5
        else:
            val = 10
        for ind in cross_index:
            viz[ind[0], ind[1]] = val
    return viz
###########################################################
def visualize_vessel_diameter_debug(edge_labels, segment_number, seg, im, use_label = False, pad = True):
    if pad == True:
        pad_size = 25
        edge_labels = np.pad(edge_labels,pad_size)
        seg = np.pad(seg, pad_size)
        im = np.pad(im,pad_size)
    segment = np.zeros_like(edge_labels).astype(np.uint8)
    segment[edge_labels==segment_number] = 1
    segment_median = segment_midpoint(segment)

    vx,vy = tangent_slope(segment, segment_median)
    bx,by = crossline_slope(vx,vy)
    
    cross_length = find_crossline_length(bx,by, segment_median, seg)
    all_cross_vals =[]
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
            all_cross_vals.append(cross_vals)

        diameter.append(diam)
    diameter, is_outlier = detect_diameter_outliers(diameter, tolerance = 0.5)
    if diameter:
        mean_diameter = np.mean(diameter)
    else:
        mean_diameter = 0
    viz = remove_outlier_crosslines(edge_labels, segment_number, seg, im, is_outlier, cross_length)
    
    if pad == True:
        im_shape = edge_labels.shape
        viz = viz[pad_size:im_shape[0]-pad_size,pad_size:im_shape[1]-pad_size]
    return diameter, mean_diameter, viz, all_cross_vals   


def whole_anatomy_diameter(im, seg, edge_labels, minimum_length = 25, pad_size = 50):
    unique_edges = np.unique(edge_labels)
    unique_edges = np.delete(unique_edges,0)
    
    edge_label_pad = np.pad(edge_labels,pad_size)
    seg_pad = np.pad(seg, pad_size)
    im_pad = np.pad(im,pad_size)
    full_viz = np.zeros_like(seg_pad)
    mean_diameters = []
    all_diameters = []
    for i in unique_edges:
        seg_length = len(np.argwhere(edge_label_pad == i))
        if seg_length>minimum_length:
            diam_list, temp_diam, temp_viz = visualize_vessel_diameter(edge_label_pad, i, seg_pad, im_pad)
            mean_diameters.append(temp_diam)
            all_diameters.append(diam_list)
            full_viz = full_viz + temp_viz
    im_shape = edge_label_pad.shape
    full_viz_no_pad = full_viz[pad_size:im_shape[0]-pad_size,pad_size:im_shape[1]-pad_size]

    return full_viz_no_pad, diameters

def visualize_vessel_diameter(edge_labels, segment_number, seg, im, use_label = False, pad = True):
    if pad == True:
        pad_size = 25
        edge_labels = np.pad(edge_labels,pad_size)
        seg = np.pad(seg, pad_size)
        im = np.pad(im,pad_size)
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==segment_number] = 1
    segment_median = segment_midpoint(segment)

    vx,vy = tangent_slope(segment, segment_median)
    bx,by = crossline_slope(vx,vy)
    
    viz = np.zeros_like(seg)
    cross_length = find_crossline_length(bx,by, segment_median, seg)
    
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

def segment_midpoint(segment):
    endpoint_index, segment_endpoints = find_endpoints(segment)
    first_endpoint = endpoint_index[0][0], endpoint_index[1][0]
    segment_indexes = np.argwhere(segment==1)
        
    distances = []
    for i in range(len(segment_indexes)):
        this_pt = segment_indexes[i][0], segment_indexes[i][1]
        distances.append(distance.chebyshev(first_endpoint, this_pt))
    sort_indexes = np.argsort(distances)
    sorted_distances = sorted(distances)
    median_val = np.median(sorted_distances)
    dist_from_median = abs(sorted_distances-median_val)
    median_distance= np.where(dist_from_median == np.min(dist_from_median))[0][0]
    segment_median = segment_indexes[median_distance]
    segment_median = segment_median.flatten()
    return segment_median
    
def tangent_slope(segment, point):
    point = point.flatten()
    crop_im = segment[point[0]-5:point[0]+5,point[1]-5:point[1]+5]
    crop_inds = np.transpose(np.where(crop_im))
    line = cv2.fitLine(crop_inds,cv2.DIST_L2,0,0.1,0.1)
    vx, vy = line[0], line[1]
    return vx, vy

def crossline_slope(vx,vy):
    bx = -vy
    by = vx
    return bx,by

def make_crossline(vx,vy,point,length):
    xlen = vx*length/2
    ylen = vy*length/2
    
    x1 = int(np.round(point[0]-xlen))
    x2 = int(np.round(point[0]+xlen))
    
    y1 = int(np.round(point[1]-ylen))
    y2 = int(np.round(point[1]+ylen))
    
    rr, cc = line(x1,y1,x2,y2)
    cross_index = []
    for r,c in zip(rr,cc):
        cross_index.append([r,c])
    coords = x1,x2,y1,y2
    
    return coords, cross_index


def plot_crossline(im, cross_index, bright = False):
    if bright == True:
        val = 250
    else:
        val = 5
    out = np.zeros_like(im)
    for i in cross_index:
        out[i[0], i[1]] = val
    return out

def find_crossline_length(vx,vy,point,im):
    distance = 5
    diam = 0
    im_size = im.shape[0]
    while diam == 0:
        distance +=5
        coords, cross_index = make_crossline(vx,vy,point,distance)
        if all(i<im_size for i in coords):
            seg_val = []
            for i in cross_index:
                seg_val.append(im[i[0], i[1]])
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
        
def crossline_intensity(cross_index, im, plot = False):
    cross_vals = []
    for i in cross_index:
        cross_vals.append(im[i[0], i[1]])
    if plot == True:
        inds = list(range(len(cross_vals)))
        plt.figure()
        plt.plot(inds, cross_vals)
    return cross_vals

def label_diameter(cross_vals):
    steps = np.where(np.roll(cross_vals,1)!=cross_vals)[0]
    if steps.size>0:
        if steps[0] == 0:
            steps = steps[1:]
        num_steps = len(steps)
        if num_steps == 2:
            diam = abs(steps[1]-steps[0])
        else:
            diam = 0
    else:
        diam = 0
    return diam

def fwhm_diameter(cross_vals):
    peak = np.max(cross_vals)
    half_max = np.round(peak/2)
    
    peak_ind = np.where(cross_vals == peak)[0][0]
    before_peak = cross_vals[0:peak_ind]
    after_peak = cross_vals[peak_ind+1:]
    try:
        hm_before = np.argmin(np.abs(before_peak - half_max))
        hm_after = np.argmin(np.abs(after_peak - half_max))
    
        # +2 added because array indexing begins at 0 twice    
        diameter = (hm_after+peak_ind) - hm_before +2
    except:
        diameter = 0
    return diameter

def fwhm_diameter_outlier(cross_vals):
    peak = np.max(cross_vals)
    half_max = np.round(peak/2)
    
    peak_ind = np.where(cross_vals == peak)[0][0]
    before_peak = cross_vals[0:peak_ind]
    after_peak = cross_vals[peak_ind+1:]
    try:
        hm_before = np.argmin(np.abs(before_peak - half_max))
        hm_after = np.argmin(np.abs(after_peak - half_max))
    
        # +2 added because array indexing begins at 0 twice    
        diameter = (hm_after+peak_ind) - hm_before +2
    except:
        diameter = 0
    return diameter

def plot_cross_vals(cross_vals):
    plt.figure()
    plt.plot(range(1,len(cross_vals)+1),cross_vals)

