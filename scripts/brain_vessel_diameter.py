#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 07:26:41 2021

brain vessel diameter

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

wt_path = '/home/sean/Documents/suchit_wt_projections/'
wt_names = ['emb9']
wt_ims = []
wt_seg = []
for im_name in wt_names:
    im = cv2.imread(wt_path+im_name+'.png',0)
    wt_ims.append(im)
    wt_seg.append(vm.brain_seg(im))
    
im = wt_ims[0]
seg = wt_seg[0]

im_crop = vm.crop_brain_im(im)
seg_crop= vm.crop_brain_im(seg)

skel = skeletonize(seg_crop)
edges, bp = vm.find_branchpoints(skel)

vm.overlay_segmentation(im_crop,skel*100+seg_crop)

_, edge_labels = cv2.connectedComponents(edges)

this_seg = 29
segment = np.zeros_like(edge_labels)
segment[edge_labels==this_seg] = 1

segment_median = segment_midpoint(segment)
distance_im = segment_distance(segment)

vx, vy = tangent_slope(segment, segment_median)
bx, by = crossline_slope(vx, vy)
coords, cross_index = make_crossline(bx,by,segment_median, 20)

cross_vals = crossline_intensity(cross_index, im_crop, plot=True)
cross_vals_seg = crossline_intensity(cross_index, seg_crop, plot=True)

##################################################################
# Full vessel

this_seg = 29
segment = np.zeros_like(edge_labels)
segment[edge_labels==this_seg] = 1

diameter, mean_diameter = vessel_diameter(edge_labels, this_seg, seg_crop)

diameter, mean_diameter, viz = visualize_vessel_diameter(edge_labels, this_seg, seg_crop)

overlay = seg_crop+skel*50+viz*20
vm.overlay_segmentation(im_crop,overlay)

def visualize_vessel_diameter(edge_labels, segment_number, seg):
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==this_seg] = 1
    segment_median = segment_midpoint(segment)

    vx,vy = tangent_slope(segment, segment_median)
    bx,by = crossline_slope(vx,vy)
    
    cross_length = find_crossline_length(bx,by, segment_median, seg)
    
    viz = np.zeros_like(seg)
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
        for i in cross_index:
            viz[i[0], i[1]] = val
        diameter.append(diam)
    diameter = [i for i in diameter if i != 0]
    mean_diameter = np.mean(diameter)
    
    return diameter, mean_diameter, viz

def vessel_diameter(edge_labels, segment_number, seg):
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==this_seg] = 1
    segment_median = segment_midpoint(segment)
    
    vx,vy = tangent_slope(segment, segment_median)
    bx,by = crossline_slope(vx,vy)
    
    cross_length = find_crossline_length(bx,by, segment_median, seg)
    
    diameter = []
    segment_inds = np.argwhere(segment)
    for i in range(10,len(segment_inds),10):
        this_point = segment_inds[i]
        vx,vy = tangent_slope(segment, this_point)
        bx,by = crossline_slope(vx,vy)
        _, cross_index = make_crossline(bx,by, this_point, cross_length)
        cross_vals = crossline_intensity(cross_index,seg)
        diameter.append(label_diameter(cross_vals))
    diameter = [i for i in diameter if i != 0]
    mean_diameter = np.mean(diameter)
    
    return diameter, mean_diameter

test_point = segment_inds[30]
vx,vy = tangent_slope(segment, test_point)
bx,by = crossline_slope(vx,vy)
_, cross_index = make_crossline(bx,by, test_point, cross_length)
cross_vals = crossline_intensity(cross_index,seg)
temp_d = label_diameter(cross_vals)

#diam = []
#for i in range(10,len(segment_inds),10):
#    this_point = segment_inds[i]
#    vx,vy = tangent_slope(segment, this_point)
#    bx,by = crossline_slope(vx,vy)
#    _, cross_index = make_crossline(bx,by, this_point, 30)
#    cross_vals = crossline_intensity(cross_index,seg_crop)
#    diam.append(label_diameter(cross_vals))
#    
#diam_overlay = multi_diam*100+seg_crop
#vm.overlay_segmentation(im_crop,diam_overlay, alpha = 0.5)

#multi_diam_color_adj = multi_diam.copy()
#multi_diam_color_adj = multi_diam_color_adj*100
#multi_diam_color_adj[0]= 1
#
#vm.overlay_segmentation(im_crop, multi_diam_color_adj, alpha = 0.9)
#
#multi_diam_skel = multi_diam*200
#multi_diam_skel = multi_diam_skel + skel*100

def overlay_seg(im,label, alpha = 0.5, contrast_adjust = False):
    if contrast_adjust:
        im = contrast_stretch(im)
        im = preprocess_seg(im)
    masked = np.ma.masked_where(label == 0, label)
    plt.figure()
    plt.imshow(im, 'gray', interpolation = 'none')
    plt.imshow(masked, 'summer', interpolation = 'none', alpha = alpha)
    plt.show()

overlay_seg(im_crop, multi_diam_skel, alpha = 0.8)

plt.imshow(edge_labels)

#######################################################################
def segment_distance(segment):
    segment_endpoints = vm.find_endpoints(segment)
    endpoint_index = np.where(segment_endpoints)
    first_endpoint = endpoint_index[0][0], endpoint_index[1][0]
    segment_indexes = np.argwhere(segment==1)
        
    distances = []
    for i in range(len(segment_indexes)):
        this_pt = segment_indexes[i][0], segment_indexes[i][1]
        distances.append(distance.chebyshev(first_endpoint, this_pt))
    distance_im =np.zeros_like(segment)
    for i in range(len(distances)):
        distance_im[segment_indexes[i][0], segment_indexes[i][1]]=distances[i]
    return distance_im

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
    while diam == 0:
        distance +=5
        _, cross_index = make_crossline(vx,vy,point,distance)
        seg_val = []
        for i in cross_index:
            seg_val.append(im[i[0], i[1]])
        steps = np.where(np.roll(seg_val,1)!=seg_val)[0]
        num_steps = len(steps)
        if num_steps == 2:
            diam = abs(steps[1]-steps[0])
        if distance >100:
            break
    length = diam*1.5
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
    if steps[0] == 0:
        steps = steps[1:]
    num_steps = len(steps)
    if num_steps == 2:
        diameter = abs(steps[1]-steps[0])
    else:
        diameter = 0
    return diameter

######################################################################
######################################################################
    
def vessel_diameter_verbose(edge_labels, segment_number, seg):
    segment = np.zeros_like(edge_labels)
    segment[edge_labels==this_seg] = 1
    
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
        seg_val = []
        for i in cross_index:
            seg_val.append(im[i[0], i[1]])
        steps = np.where(np.roll(seg_val,1)!=seg_val)[0]
        if steps[0] == 0:
            steps = steps[1:]
        num_steps = len(steps)
        if num_steps == 2:
            diam = abs(steps[1]-steps[0])
        if dist >100:
            break
    length = diam*1.5
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
        ###############################################
        
        cross_vals = crossline_intensity(cross_index,seg)
        
        cross_vals = []
        for i in cross_index:
            cross_vals.append(im[i[0], i[1]])
        ###########################################
        steps = np.where(np.roll(cross_vals,1)!=cross_vals)[0]
        if steps[0] == 0:
            steps = steps[1:]
        num_steps = len(steps)
        if num_steps == 2:
            diam = abs(steps[1]-steps[0])
        else:
            diam = 0
        
        ############################################
        if diam == 0:
            val = 5
        else:
            val = 10
        for i in cross_index:
            viz[i[0], i[1]] = val
        diameter.append(diam)
    diameter = [i for i in diameter if i != 0]
    mean_diameter = np.mean(diameter)
    
    return diameter, mean_diameter, viz



