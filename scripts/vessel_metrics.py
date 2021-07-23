#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 07:51:19 2021

Segment tools - tools for extracting metrics from a binary mask
of blood vessel image

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


def fill_holes(label_binary, hole_size):
    label_inv = np.bitwise_not(label_binary)
    label_inv[label_inv<255] = 0
    _, inverted_labels, stats, _ = cv2.connectedComponentsWithStats(label_inv)
    
    vessel_sizes = stats[:,4]
    small_vessel_inds = np.argwhere(vessel_sizes<hole_size)
    
    for v in small_vessel_inds:
        inverted_labels[inverted_labels == v] = 0
        
    label_mask = np.zeros_like(label_inv)
    label_mask[inverted_labels==0] = 1 
    
    skel = skeletonize(label_mask)
    return skel, label_mask


def tortuosity(edge_labels, end_points):
    endpoint_labeled = edge_labels*end_points
    unique_labels = np.unique(edge_labels)
    unique_labels = unique_labels[1:]
    tortuosity = []
    for u in unique_labels:
        this_segment = np.zeros_like(edge_labels)
        this_segment[edge_labels == u] = 1
        
        end_inds = np.argwhere(endpoint_labeled == u)
#        try:
            # draw a line between the end points, count the pixels
        end_point_line = draw.line(end_inds[0,0],end_inds[0,1],end_inds[1,0],end_inds[1,1])
        endpoint_distance = np.max(np.shape(end_point_line))
#        except:
#            print(u)
        segment_length = np.sum(this_segment)
        tortuosity.append(endpoint_distance/segment_length)
    return tortuosity, unique_labels

def remove_small_segments(edge_labels, minimum_length):
    unique_labels = np.unique(edge_labels)
    unique_labels = unique_labels[1:]
    for u in unique_labels:
        this_seg_count = np.shape(np.argwhere(edge_labels==u))[0]
        if this_seg_count < minimum_length:
            edge_labels[edge_labels==u] = 0
    edges = np.zeros_like(edge_labels)
    edges[edge_labels>0] = 1
    edges = edges.astype(np.uint8)
    _, edge_labels_new = cv2.connectedComponents(edges, connectivity = 8)
    return edge_labels_new, edges


def find_branchpoints(skel):
    skel_binary = np.zeros_like(skel)
    skel_binary[skel>0] = 1
    skel_index = np.argwhere(skel_binary == True)
    tile_sum=[]
    neighborhood_image = np.zeros(skel.shape)
    for i,j in skel_index:
        this_tile = skel_binary[i-1:i+2,j-1:j+2]
        tile_sum.append(np.sum(this_tile))
        neighborhood_image[i,j] = np.sum(this_tile)
    branch_points_messy = np.zeros_like(neighborhood_image)
    branch_points_messy[neighborhood_image>3] = 1
    branch_points_messy = branch_points_messy.astype(np.uint8)

    branch_points = branch_points_messy
    edges = np.zeros_like(branch_points_messy)
    edges = skel.astype(np.uint8) - branch_points_messy
    
    return edges, branch_points

def find_endpoints(edges):
    edge_binary = np.zeros_like(edges)
    edge_binary[edges>0] = 1
    edge_index = np.argwhere(edge_binary == True)
    tile_sum=[]
    neighborhood_image = np.zeros(edges.shape)

    for i,j in edge_index:
        this_tile = edge_binary[i-1:i+2,j-1:j+2]
        tile_sum.append(np.sum(this_tile))
        neighborhood_image[i,j] = np.sum(this_tile)
    end_points = np.zeros_like(neighborhood_image)
    end_points[neighborhood_image == 2] = 1
    return end_points

def vessel_length(edge_labels):
    unique_segments, segment_counts = np.unique(edge_labels, return_counts = True)
    unique_segments = unique_segments[1:]
    segment_counts = segment_counts[1:]
    return unique_segments, segment_counts

def normalize_contrast(image):
    img_norm = image/np.max(image)
    img_adj = np.floor(img_norm*255)
    return img_adj

def find_terminal_segments(skel, edge_labels):
    skel[skel>0]=1
    skel_index = np.argwhere(skel == True)
    tile_sum=[]
    neighborhood_image = np.zeros(skel.shape)

    for i,j in skel_index:
        this_tile = skel[i-1:i+2,j-1:j+2]
        tile_sum.append(np.sum(this_tile))
        neighborhood_image[i,j] = np.sum(this_tile)
    terminal_points = np.zeros_like(neighborhood_image)
    terminal_points[neighborhood_image == 2] = 1
    
    terminal_segments = np.zeros_like(terminal_points)
    unique_labels = np.unique(edge_labels)
    unique_labels = unique_labels[1:]
    
    for u in unique_labels:
        this_segment = np.zeros_like(terminal_points)
        this_segment[edge_labels == u] = 1
        overlap = this_segment + terminal_points
        if len(np.argwhere(overlap>1)):
            terminal_segments[edge_labels == u] = 1
    return terminal_segments

def preprocess_czi(input_directory,file_name):
    with CziFile(input_directory + file_name) as czi:
        image_arrays = czi.asarray()

    image = np.squeeze(image_arrays)
    im_channel = image[0,:,:,:]
    im_channel = normalize_contrast(im_channel)
    return im_channel
    
def czi_projection(volume,axis):
    projection = np.max(volume, axis = axis)
    return projection

def segment_chunk(segment_number, edge_labels, volume):
    segment_inds = np.argwhere(edge_labels == segment_number)
    this_segment = np.zeros_like(edge_labels)
    this_segment[edge_labels == segment_number] = 1
    this_segment = this_segment.astype(np.uint8)
    end_points = find_endpoints(this_segment)
    end_inds = np.argwhere(end_points>0)
    mid_point = np.round(np.mean(end_inds, axis = 0)).astype(np.uint64)
    tile_size = np.abs(np.array([np.shape(volume)[0],end_inds[0,0]-end_inds[1,0], end_inds[0,1]-end_inds[1,1]])*1.5).astype(np.uint8)
    chunk = volume[:,mid_point[0]-tile_size[0]:mid_point[0]+tile_size[0], mid_point[1]-tile_size[1]:mid_point[1]+tile_size[1]]
    return chunk

def remove_small_objects(label, size_thresh):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(label.astype(np.uint8))
    
    object_sizes = stats[:,4]
    small_obj_inds = np.argwhere(object_sizes<size_thresh)
    
    for v in small_obj_inds:
        labels[labels == v] = 0
    
    output = np.zeros_like(labels)
    output[labels>0] = 1
        
    return output

def segment_vessels(image,k = 12, hole_size = 500, ditzle_size = 750, bin_thresh = 2):
    image = cv2.medianBlur(image.astype(np.uint8),7)
    image, background = subtract_background_rolling_ball(image, 400, light_background=False,
                                         use_paraboloid=False, do_presmooth=True)
    im_vector = image.reshape((-1,)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(im_vector,k,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = center.astype(np.uint8)
    label_im = label.reshape((image.shape))
    seg_im = np.zeros_like(label_im)
    for i in np.unique(label_im):
        seg_im = np.where(label_im == i, center[i],seg_im)
    
    _, seg_im = cv2.threshold(seg_im.astype(np.uint16), bin_thresh, 255, cv2.THRESH_BINARY)
    
    _, seg_im = fill_holes(seg_im.astype(np.uint8),hole_size)
    seg_im = remove_small_objects(seg_im,ditzle_size)
    
    return seg_im

def preprocess_seg(image):
    image = cv2.medianBlur(image.astype(np.uint8),7)
    image, background = subtract_background_rolling_ball(image, 400, light_background=False,
                                                            use_paraboloid=False, do_presmooth=True)
    return image

def sliding_window(volume,thickness):
    num_slices = np.shape(volume)[0]
    out_slices = num_slices-thickness
    output = np.zeros([out_slices, np.shape(volume)[1], np.shape(volume)[2]])
    for i in range(0,out_slices):
        im_chunk = volume[i:i+thickness,:,:]
        output[i,:,:] = np.max(im_chunk,axis = 0)
    return output

def jaccard(ground_truth, label):
    ground_truth_binary = np.zeros_like(ground_truth)
    label_binary = np.zeros_like(label)
    
    ground_truth_binary[ground_truth>0] = 1
    label_binary[label>0] = 1
    
    intersection = np.sum(np.logical_and(ground_truth_binary, label_binary))
    union = np.sum(np.logical_or(ground_truth_binary, label_binary))
    
    jacc = round(intersection/union,2)
    return jacc

def signal_to_noise(image):
    step_size = [round(z/8) for z in np.shape(image)]
    mid_point = [round(z/2) for z in np.shape(image)]

    roi = image[mid_point[0]-step_size[0]:mid_point[0]+step_size[0],mid_point[1]-step_size[1]:mid_point[1]+step_size[1]]
    
    px_mean = np.mean(roi)
    px_std = np.std(roi)
    
    snr = px_mean/px_std

    return snr

def clahe(im, tiles = (16,16), clip_lim = 40):
    cl = cv2.createCLAHE(clipLimit = clip_lim, tileGridSize = tiles)
    im = im.astype(np.uint16)
    im = cl.apply(im)
    im = normalize_contrast(im)
    return im

def cal(ground_truth, label):
    ground_truth_binary = np.zeros_like(ground_truth)
    label_binary = np.zeros_like(label)
    
    ground_truth_binary[ground_truth>0] = 1
    label_binary[label>0] = 1
    
    num_labels_gt, labels_gt = cv2.connectedComponents(ground_truth_binary, connectivity = 8)
    num_labels_l, labels_l = cv2.connectedComponents(label_binary, connectivity = 8)
    
    connectivity = round(1 - np.min([1,np.abs(num_labels_gt-num_labels_l)/np.sum(ground_truth_binary)]),2)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilation_gt = cv2.dilate(ground_truth_binary,kernel)
    dilation_l = cv2.dilate(label_binary,kernel)
    
    dilated_label_union = np.logical_and(ground_truth_binary == 1, dilation_l == 1)
    dilated_gt_union = np.logical_and(label_binary == 1, dilation_gt == 1)
    
    area_numerator = np.sum(np.logical_or(dilated_label_union, dilated_gt_union))
    area_denominator = np.sum(np.logical_or(label_binary,ground_truth_binary))
    
    area = round(area_numerator/area_denominator,2)
    
    gt_skeleton = skeletonize(ground_truth_binary)
    l_skeleton = skeletonize(label_binary)
    
    label_skel_int = np.logical_and(l_skeleton,dilation_gt)
    gt_skel_int = np.logical_and(gt_skeleton,dilation_l)
    
    length_numerator = np.sum(np.logical_or(label_skel_int,gt_skel_int))
    length_denominator = np.sum(np.logical_or(l_skeleton,gt_skeleton))
    
    length = round(length_numerator/length_denominator,2)
    
    return length, area,  connectivity

def seg_holes(label):
    label_inv = np.zeros_like(label)
    label_inv[label == 0] = 1
    label_inv = label_inv.astype(np.uint8)
    _, labelled_holes, stats, _ = cv2.connectedComponentsWithStats(label_inv)
    return labelled_holes, label_inv, stats


#########################################################

def crossline_endpoints(label,start,slope):    
    current_point = start
    current_label_val = label[current_point[0],current_point[1]]
    while current_label_val:
        current_x = current_point[0]-1
        current_y = current_point[1]-slope
        current_point = np.int(round(current_x)), np.int(round(current_y))
        current_label_val = label[current_point[0],current_point[1]]
    end_point1 = current_point
    
    current_point = start
    current_label_val = label[current_point[0],current_point[1]]
    while current_label_val:
        current_x = current_point[0]+1
        current_y = current_point[1]+slope
        current_point = np.int(round(current_x)), np.int(round(current_y))
        current_label_val = label[current_point[0],current_point[1]]
    end_point2 = current_point
    return end_point1, end_point2

def find_segment_crossline_length(label,start,slope):
    current_point = start
    current_label_val = label[current_point[0],current_point[1]]
    while current_label_val:
        current_x = current_point[0]-1
        current_y = current_point[1]-slope
        current_point = current_x, current_y
        current_label_val = label[np.int(np.round(current_point[0])),np.int(np.round(current_point[1]))]
    end_point = np.int(np.round(current_x)), np.int(np.round(current_y))
    vessel_radius = distance.chebyshev(end_point,start)
    cross_thickness = vessel_radius*1.5
    return cross_thickness

def distance_along_line(point,slope,distance):
    x_dist = np.sqrt(distance**2/(slope**2+1))
    y_dist = x_dist*slope
    
    x_dist = np.round(x_dist)
    y_dist = np.round(y_dist)
    
    return x_dist, y_dist
    
def calculate_crossline(point, slope, x_dist, y_dist):
    x1 = np.int(point[0]-x_dist)
    y1 = np.int(point[1]-y_dist)
    
    x2 = np.int(point[0]+x_dist)
    y2 = np.int(point[1]+y_dist)
    
    cross_index = list(bresenham(x1,y1,x2,y2))
    
    return cross_index

def vessel_diameter(label, segment):
    segment_endpoints = find_endpoints(segment)
    endpoint_index = np.where(segment_endpoints)
    first_endpoint = endpoint_index[0][0], endpoint_index[1][0]
    segment_indexes = np.argwhere(segment==1)
    
    distances = []
    for i in range(len(segment_indexes[0])):
        this_pt = segment_indexes[0][i], segment_indexes[1][i]
        distances.append(distance.chebyshev(first_endpoint, this_pt))
    sort_indexes = np.argsort(distances)
    sorted_distances = sorted(distances)
    median_index = np.where(sorted_distances == np.median(sorted_distances))[0][0]
    segment_median = sort_indexes[median_index]
    
    start_pt = segment_indexes[sort_indexes[median_index-3]]
    end_pt = segment_indexes[sort_indexes[median_index+3]]
    median_pt = segment_indexes[sort_indexes[median_index]]
    slope = (start_pt[0]-end_pt[0])/(start_pt[1]-end_pt[1])
    cross_slope = -1/slope

    cross_length = find_segment_crossline_length(label, median_pt,cross_slope)
    
    diameters = []
    for i in range(10,len(sort_indexes)-3,10):
        print(i)
        start_pt = segment_indexes[sort_indexes[i-3]]
        end_pt = segment_indexes[sort_indexes[i+3]]
        mid_pt = segment_indexes[sort_indexes[i]]
        slope = (start_pt[0]-end_pt[0])/(start_pt[1]-end_pt[1])

        if slope != 0:
            cross_slope = -1/slope
    
            x_dist, y_dist = distance_along_line(mid_pt, cross_slope, cross_length)
            cross_index = calculate_crossline(mid_pt, cross_slope, x_dist, y_dist)
        else:
            x_dist = 0
            y_dist = cross_length
            
            x1 = np.int(mid_pt[0]-x_dist)
            y1 = np.int(mid_pt[1]-y_dist)
    
            x2 = np.int(mid_pt[0]+x_dist)
            y2 = np.int(mid_pt[1]+y_dist)
    
            cross_index = list(bresenham(x1,y1,x2,y2))
        
        label_intensity = []
        for j in cross_index:
            label[j[0],j[1]] = 3
            label_intensity.append(label[j[0],j[1]])
        this_diameter = np.sum(label_intensity)
        diameters.append(this_diameter)
    return diameters, label, label_intensity


#####################################################################
# DEPRECATED 
    
def czi2mip(input_directory):
    file_list = os.listdir(input_directory)
    for i in file_list:
        output_name = i.replace(" ","_")
        output_name = output_name.replace('.czi','')
        with CziFile(input_directory + i) as czi:
            image_arrays = czi.asarray()
    
        image = np.squeeze(image_arrays)
        
        if image.ndim==3:
            image = image[np.newaxis,:,:,:]
        
        for c in range(image.shape[0]):
            output_name_channel = output_name + '_ch' + str(c) + '.png'
            projection = np.max(image[c,:,:,:], axis = 0)
            img_norm = projection/np.max(projection)
            img_adj = np.floor(img_norm*255)        
        cv2.imwrite(output_directory + output_name_channel, projection)


def segment_viewer(segment_number, edge_labels, image):
    segment_inds = np.argwhere(edge_labels == segment_number)
    this_segment = np.zeros_like(edge_labels)
    this_segment[edge_labels == segment_number] = 1
    this_segment = this_segment.astype(np.uint8)
    end_points = find_endpoints(this_segment)
    end_inds = np.argwhere(end_points>0)
    mid_point = np.round(np.mean(end_inds, axis = 0)).astype(np.uint64)
    tile_size = np.abs(np.array([end_inds[0,0]-end_inds[1,0], end_inds[0,1]-end_inds[1,1]])*1.5).astype(np.uint8)
    tile = image[mid_point[0]-tile_size[0]:mid_point[0]+tile_size[0], mid_point[1]-tile_size[1]:mid_point[1]+tile_size[1]]
    edge_tile = edge_labels[mid_point[0]-tile_size[0]:mid_point[0]+tile_size[0], mid_point[1]-tile_size[1]:mid_point[1]+tile_size[1]]
    plt.figure(); 
    plt.subplot(1,2,1)
    plt.imshow(tile)
    plt.subplot(1,2,2)
    plt.imshow(edge_tile)
    
def reslice_image(image,thickness):
    num_slices = np.shape(image)[0]
    out_slices = np.ceil(num_slices/thickness).astype(np.uint16)
    output = np.zeros([out_slices, np.shape(image)[1], np.shape(image)[2]])
    count = 0
    for i in range(0,num_slices, thickness):
        if i+thickness<num_slices:
            im_chunk = image[i:i+thickness,:,:]
        else:
            im_chunk = image[i:,:,:]
        
        output[count,:,:] = np.max(im_chunk,axis = 0)
        count+=1
    return output