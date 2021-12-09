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

visualize_branch_points = False
if visualize_branch_points:
    kernel = np.ones([5,5])
    dilated_bp = cv2.dilate(bp,kernel, iterations = 1)
    vm.overlay_segmentation(im_crop, dilated_bp*20,alpha = 1)
_, edge_labels = cv2.connectedComponents(edges)

unique_edges = np.unique(edge_labels)
unique_edges = np.delete(unique_edges,0)

# zero padding
pad_size = 50
edge_label_pad = np.pad(edge_labels,pad_size)
seg_pad = np.pad(seg_crop, pad_size)

minimum_length = 25
full_viz = np.zeros_like(seg_pad)
diameters = []
for i in unique_edges:
    seg_length = len(np.argwhere(edge_label_pad == i))
    if seg_length>minimum_length:
        print(i)
        _, temp_diam, temp_viz = visualize_vessel_diameter(edge_label_pad, i, seg_pad)
        diameters.append(temp_diam)
        full_viz = full_viz + temp_viz
        
plt.imshow(full_viz)
im_shape = edge_label_pad.shape
full_viz_no_pad = full_viz[pad_size:im_shape[0]-pad_size,pad_size:im_shape[1]-pad_size]

overlay = seg_crop + full_viz_no_pad + skel
vm.overlay_segmentation(im_crop, overlay)


def whole_anatomy_diameter(seg, edge_labels, minimum_length = 25, pad_size = 50): 
    unique_edges = np.unique(edge_labels)
    unique_edges = np.delete(unique_edges,0)
    
    edge_label_pad = np.pad(edge_labels,pad_size)
    seg_pad = np.pad(seg, pad_size)
    full_viz = np.zeros_like(seg_pad)
    diameters = []
    for i in unique_edges:
        seg_length = len(np.argwhere(edge_label_pad == i))
        if seg_length>minimum_length:
            _, temp_diam, temp_viz = visualize_vessel_diameter(edge_label_pad, i, seg_pad)
            diameters.append(temp_diam)
            full_viz = full_viz + temp_viz
    im_shape = edge_label_pad.shape
    full_viz_no_pad = full_viz[pad_size:im_shape[0]-pad_size,pad_size:im_shape[1]-pad_size]
    
    return full_viz_no_pad, diameters


########################################################################
    
new_im = 'emb2'
im2 = cv2.imread(wt_path+new_im+'.png',0)
seg2 = vm.brain_seg(im2, sato_thresh = 40)

im_crop2 = vm.crop_brain_im(im2)
seg_crop2 = vm.crop_brain_im(seg2)

vm.overlay_segmentation(im_crop2, seg_crop2)

skel2 = skeletonize(seg_crop2)
edges2 , bp = vm.find_branchpoints(skel2)

_, edge_labels2 = cv2.connectedComponents(edges2)

full_viz2, diameters2 = whole_anatomy_diameter(seg_crop2, edge_labels2)

overlay2 = seg_crop2 + full_viz2 + skel2
vm.overlay_segmentation(im_crop2, overlay2)

######################################################################
seg_num = 79
im_contrast = vm.contrast_stretch(im_crop2)
plt.imshow(im_crop2)
im_preproc = vm.preprocess_seg(im_contrast)
plt.imshow(im_preproc)
im_sato = sato(im_preproc, sigmas = range(1,10,2), mode = 'reflect', black_ridges = False)
plt.imshow(im_sato)
im_sato_norm = np.round(im_sato/np.max(im_sato)*255).astype(np.uint8)
    
segment = np.zeros_like(edge_labels2)
segment[edge_labels2==seg_num] = 1
segment_median = segment_midpoint(segment)

vx,vy = tangent_slope(segment, segment_median)
bx,by = crossline_slope(vx,vy)

viz = np.zeros_like(im_crop2)
cross_length = find_crossline_length(bx,by, segment_median, seg_crop2)

_, cross_index = make_crossline(bx,by,segment_median, cross_length)
cross_vals_seg = crossline_intensity(cross_index, seg_crop2)
cross_vals_raw = crossline_intensity(cross_index, im_crop2)
cross_vals_contrast = crossline_intensity(cross_index, im_contrast)
cross_vals_smoothed = crossline_intensity(cross_index, im_preproc)
cross_vals_sato= crossline_intensity(cross_index, im_sato_norm)

plt.figure()
x = range(len(cross_index))
plt.plot(x,cross_vals_seg)

plt.figure()
plt.plot(x,cross_vals_raw)

plt.figure()
plt.plot(x,cross_vals_contrast)

plt.figure();
plt.plot(x,cross_vals_smoothed)

plt.figure()
plt.plot(x,cross_vals_sato)

for ind in cross_index:
    viz[ind[0], ind[1]] = 200
    
vm.overlay_segmentation(im_crop2, viz)

########################################################################
ret_path = '/home/sean/Downloads/labels-ah/'
ret_name = 'im0001.ah.ppm'

ret_im = cv2.imread(ret_path+ret_name,0)
ret_im[ret_im>0] = 1
ret_skel = skeletonize(ret_im)

vm.overlay_segmentation(ret_im, ret_skel)

ret_edges , bp = vm.find_branchpoints(ret_skel)

_, ret_edge_labels = cv2.connectedComponents(ret_edges)

ret_viz, ret_diam = whole_anatomy_diameter(ret_im, ret_edge_labels)

ret_overlay = ret_viz + ret_skel
vm.overlay_segmentation(ret_im, ret_overlay)

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
    if diameter:
        mean_diameter = np.mean(diameter)
    else:
        mean_diameter = 0
    
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


    
########################################################################
    
edge_label_test = deepcopy(edge_labels)
segment_number = 9698
seg_test = deepcopy(seg_crop2)

edge_label_test = np.pad(edge_label_test,50)
seg_test = np.pad(seg_test,50)

segment = np.zeros_like(edge_label_test)
segment[edge_label_test==segment_number] = 1

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
im_size = seg_test.shape[0]
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
        print('coords within image bounds, dist ' + str(dist))
        seg_val = []
        for i in cross_index:
            seg_val.append(seg_test[i[0], i[1]])
        steps = np.where(np.roll(seg_val,1)!=seg_val)[0]
        if steps.size>0:
            if steps[0] == 0:
                steps = steps[1:]
            num_steps = len(steps)
            if num_steps == 2:
                diam = abs(steps[1]-steps[0])
        if dist >100:
            print('dist > 100')
            break
    else:
        print('coords not within image bounds')
        break
length = diam*2.5
################################################################
viz = np.zeros_like(seg_test)
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
        cross_vals.append(seg_test[c[0], c[1]])
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

########################################################################
########################################################################

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

