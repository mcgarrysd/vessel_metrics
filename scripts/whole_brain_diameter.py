#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:36:58 2021

Whole brain diameter

@author: sean
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
from timeit import default_timer as timer
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from scipy import stats
from scipy.spatial import distance
from skimage.draw import line
from copy import deepcopy

wt_path = '/home/sean/Documents/suchit_wt_projections/'
wt_names = 'emb9'

im = cv2.imread(wt_path+wt_names+'.png',0)
seg = vm.brain_seg(im)

im_crop = vm.crop_brain_im(im)
seg_crop= vm.crop_brain_im(seg)

skel = skeletonize(seg_crop)
edges, bp = vm.find_branchpoints(skel)

_, edge_labels = cv2.connectedComponents(edges)

unique_edges = np.unique(edge_labels)
unique_edges = np.delete(unique_edges,0)

minimum_length = 50
full_viz = np.zeros_like(im_crop)
diameters = []
for i in unique_edges:
    seg_length = len(np.argwhere(edge_labels == i))
    if seg_length>minimum_length:
        print(i)
        _, temp_diam, temp_viz = visualize_vessel_diameter(edge_labels, i, seg_crop)
        diameters.append(temp_diam)
        full_viz = full_viz + temp_viz
        
overlay = seg_crop + full_viz + skel
vm.overlay_segmentation(im_crop, overlay)



################################################################################

def visualize_vessel_diameter(edge_labels, segment_number, seg):
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
        cross_vals = crossline_intensity(cross_index,seg)
        diam = label_diameter(cross_vals)
        if diam == 0:
            val = 5
        else:
            val = 10
        for ind in cross_index:
            viz[ind[0], ind[1]] = val
        diameter.append(diam)
    diameter = [x for x in diameter if x != 0]
    mean_diameter = np.mean(diameter)
    
    return diameter, mean_diameter, viz
    
#####################################################################
    
def segment_midpoint(segment):
    segment_endpoints = vm.find_endpoints(segment)
    endpoint_index = np.where(segment_endpoints)
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
    segment_median = segment_indexes[np.where(distances == median_distance)]
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
    
    for i in cross_index:
        im[i[0], i[1]] = val
    return im

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
                seg_val.append(seg[i[0], i[1]])
            steps = np.where(np.roll(seg_val,1)!=seg_val)[0]
            if steps.size>0:
                if steps[0] == 0:
                    steps = steps[1:]
                num_steps = len(steps)
                if num_steps == 2:
                    diam = abs(steps[1]-steps[0])
            if dist >100:
                break
        else:
            break
    length = diam*2
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


########################################################################
    
  
segment_number = 53
seg = deepcopy(seg_crop)

def vessel_diameter_verbose(edge_labels, segment_number, seg):
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==segment_number] = 1
    
    #########################################################
    
    segment_endpoints = vm.find_endpoints(segment)
    endpoint_index = np.where(segment_endpoints)
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
    segment_median = segment_indexes[np.where(distances == median_distance)]
    segment_median = segment_median.flatten()

    ####################################################
    point = segment_median
    crop_im = segment[point[0]-5:point[0]+5,point[1]-5:point[1]+5]
    crop_inds = np.transpose(np.where(crop_im))
    tangent = cv2.fitLine(crop_inds,cv2.DIST_L2,0,0.1,0.1)
    vx, vy = tangent[0], tangent[1]
    bx = -vy
    by = vx
    
    #####################################################
    dist = 5
    diam = 0
    im_size = seg.shape[0]
    while diam == 0:
        dist +=5
        xlen = bx*dist/2
        ylen = by*dist/2
    
        x1 = int(np.round(point[0]-xlen))
        x2 = int(np.round(point[0]+xlen))
    
        y1 = int(np.round(point[1]-ylen))
        y2 = int(np.round(point[1]+ylen))
    
        rr, cc = line(x1,y1,x2,y2)
        cross_index = []
        for r,c in zip(rr,cc):
            cross_index.append([r,c])
        coords = x1,x2,y1,y2
        if all(i<im_size for i in coords):
            seg_val = []
            for i in cross_index:
                seg_val.append(seg[i[0], i[1]])
            steps = np.where(np.roll(seg_val,1)!=seg_val)[0]
            if step.size>0:
                if steps[0] == 0:
                    steps = steps[1:]
                num_steps = len(steps)
                if num_steps == 2:
                    diam = abs(steps[1]-steps[0])
            if dist >100:
                break
        else:
            break
    length = diam*2.5
    ################################################################
    viz = np.zeros_like(seg)
    diameter = []
    segment_inds = np.argwhere(segment)
    for i in range(10,len(segment_inds),10):
        point = segment_inds[i]
        crop_im = segment[point[0]-5:point[0]+5,point[1]-5:point[1]+5]
        crop_inds = np.transpose(np.where(crop_im))
        tangent = cv2.fitLine(crop_inds,cv2.DIST_L2,0,0.1,0.1)
        vx, vy = tangent[0], tangent[1]
        bx = -vy
        by = vx
        
        
        xlen = bx*length/2
        ylen = by*length/2
    
        x1 = int(np.round(point[0]-xlen))
        x2 = int(np.round(point[0]+xlen))
    
        y1 = int(np.round(point[1]-ylen))
        y2 = int(np.round(point[1]+ylen))
    
        rr, cc = line(x1,y1,x2,y2)
        cross_index = []
        for r,c in zip(rr,cc):
            cross_index.append([r,c])
        ###############################################
        
        cross_vals = []
        for c in cross_index:
            cross_vals.append(seg[c[0], c[1]])
        ###########################################
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
        
        ############################################
        if diam == 0:
            val = 5
        else:
            val = 10
        for c in cross_index:
            viz[c[0], c[1]] = val
        diameter.append(diam)
    diameter = [x for x in diameter if x != 0]
    mean_diameter = np.mean(diameter)
    
    return diameter, mean_diameter, viz

