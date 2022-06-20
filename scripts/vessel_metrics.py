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
from skimage.filters import meijering, hessian, frangi, sato
from skimage.draw import line
from bresenham import bresenham
from skimage.util import invert
from skimage.filters.ridges import compute_hessian_eigenvalues
import itertools
from math import dist

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
    coords = np.argwhere(end_points==1)
    return coords, end_points

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

def preprocess_czi(input_directory,file_name, channel = 0):
    with CziFile(input_directory + file_name) as czi:
        image_arrays = czi.asarray()

    image = np.squeeze(image_arrays)
    im_channel = image[channel,:,:,:]
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
    end_inds, end_points = find_endpoints(this_segment)
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

def preprocess_seg(image,ball_size = 400, median_size = 7, upper_lim = 255, lower_lim = 0):
    image, background = subtract_background_rolling_ball(image, ball_size, light_background=False,
                                                            use_paraboloid=False, do_presmooth=True)
    image = cv2.medianBlur(image.astype(np.uint8),median_size)
    image = contrast_stretch(image, upper_lim = upper_lim, lower_lim = lower_lim)
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

def contrast_stretch(image,upper_lim = 255, lower_lim = 0):
    c = np.percentile(image,5)
    d = np.percentile(image,95)
    
    stretch = (image-c)*((upper_lim-lower_lim)/(d-c))+lower_lim
    stretch[stretch<lower_lim] = lower_lim
    stretch[stretch>upper_lim] = upper_lim
    
    return stretch


def branchpoint_density(skel, label):
    _, bp = find_branchpoints(skel)
    _, bp_labels = cv2.connectedComponents(bp, connectivity = 8)
    
    skel_inds = np.argwhere(skel > 0)
    
    bp_density = []
    for i in range(0,len(skel_inds), 50):
        x = skel_inds[i][0]; y = skel_inds[i][1]
        this_tile = bp_labels[x-25:x+25,y-25:y+25]
        bp_number = len(np.unique(this_tile))-1
        bp_density.append(bp_number)
        
    bp_density = np.array(bp_density)
    bp_density[bp_density<0] = 0
    return bp_density


def overlay_segmentation(im,label, alpha = 0.5, contrast_adjust = False, im_cmap = 'gray', label_cmap = 'jet'):
    if contrast_adjust:
        im = contrast_stretch(im)
        im = preprocess_seg(im)
    masked = np.ma.masked_where(label == 0, label)
    plt.figure()
    plt.imshow(im, 'gray', interpolation = 'none')
    plt.imshow(masked, 'jet', interpolation = 'none', alpha = alpha)
    plt.show()
    
def vessel_density(im,label, num_tiles_x, num_tiles_y):
    density = np.zeros_like(im).astype(np.float16)
    density_array = []
    label[label>0] = 1
    step_x = np.round(im.shape[0]/num_tiles_x).astype(np.int16)
    step_y = np.round(im.shape[1]/num_tiles_y).astype(np.int16)
    for x in range(0,im.shape[0], step_x):
        for y in range(0,im.shape[1], step_y):
            tile = label[x:x+step_x-1,y:y+step_y-1]
            numel = tile.shape[0]*tile.shape[1]
            tile_density = np.sum(tile)/numel
            tile_val = np.round(tile_density*1000)
            density[x:x+step_x-1,y:y+step_y-1] = tile_val
            density_array.append(tile_val)
    density = density.astype(np.uint16)
    return density, density_array

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

def connect_segments(skel):
    skel = np.pad(skel,50)
    edges, bp = find_branchpoints(skel)
    _,edge_labels = cv2.connectedComponents(edges)
    
    edge_labels[edge_labels!=0]+=1
    bp_el = edge_labels+bp
    
    _, bp_labels = cv2.connectedComponents(bp)
    unique_bp = np.unique(bp_labels)
    unique_bp = unique_bp[1:]
    
    bp_list = []
    bp_connections = []
    new_edges = np.zeros_like(skel)
    new_bp = np.zeros_like(skel)
    for i in unique_bp:
        temp_bp = np.zeros_like(bp_labels)
        temp_bp[bp_labels == i] = 1
        bp_size = np.sum(temp_bp)
        if bp_size>1:
            this_bp_inds = np.argwhere(temp_bp == 1)
            
            connected_segs = []
            bp_coords = []
            for x,y in this_bp_inds:
                bp_neighbors = bp_el[x-1:x+2,y-1:y+2]
                if np.any(bp_neighbors>1):
                    connections = bp_neighbors[bp_neighbors>1].tolist()
                    connected_segs.append(connections)
                    for c in connections:    
                        bp_coords.append((x,y))
            bp_list.append(i)
            bp_connections.append(connected_segs)
            connected_segs = flatten(connected_segs)
            
            vx = []
            vy = []
            for seg in connected_segs:
                #print('segment ' + str(seg))
                temp_seg = np.zeros_like(bp_labels)
                temp_seg[edge_labels == seg] = 1
                endpoints, endpoint_im = find_endpoints(temp_seg)
                if np.size(endpoints):
                    line = cv2.fitLine(endpoints,cv2.DIST_L2,0,0.1,0.1)
                    vx.append(float(line[0]))
                    vy.append(float(line[1]))
                
            vx = np.array(vx).flatten().tolist()
            vy = np.array(vy).flatten().tolist()
            
            
            v_r = list(zip(np.round(vx,3), np.round(vy,3)))
            slope_tolerance = 0.1
            
            inds = list(range(len(v_r)))
            pair_inds = list(itertools.combinations(inds, 2))
            count = 0
            match = []
            for x,y in itertools.combinations(v_r, 2):
                if np.abs(x[0] - y[0])<slope_tolerance:
                    if np.abs(x[1]-y[1])<slope_tolerance:
                      match = pair_inds[count]
                count+=1
                
            if match:
                c1 = bp_coords[match[0]]
                c2 = bp_coords[match[1]]
                connected_pts = list(bresenham(c1[0],c1[1],c2[0],c2[1]))
                temp_edges = np.zeros_like(bp_labels)
                temp_bp = np.zeros_like(bp_labels)
                for x,y in connected_pts:
                    temp_edges[x,y] = 1
                new_edges = new_edges+temp_edges
                for x,y in this_bp_inds:
                    if temp_edges[x,y] == 0:
                        bp_neighbors = edges[x-1:x+2,y-1:y+2]
                        if np.any(bp_neighbors>0):
                            temp_bp[x,y] = 1
                new_bp = temp_bp+new_bp
            else:
                new_bp = new_bp + temp_bp
        else:
            new_bp = new_bp + temp_bp
    new_edges = new_edges + edges
    xdim, ydim = np.shape(skel)
    
    new_edges = new_edges[50:xdim-50, 50:ydim-50]
    new_bp = new_bp[50:xdim-50, 50:ydim-50]
    return new_edges, new_bp

def flatten(input_list):
    return [item for sublist in input_list for item in sublist]

def network_length(edges):
    edges[edges>0]=1
    net_length = np.sum(edges)
    return net_length

def prune_terminal_segments(skel, seg_thresh = 20):
    edges, bp = connect_segments(skel)
    _, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    _, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))
    
    terminal_segs = find_terminal_segments(skel, edge_labels)
    new_terminal_segs = np.zeros_like(terminal_segs)
    _, term_labels = cv2.connectedComponents(terminal_segs.astype(np.uint8))
    unique_labels = np.unique(term_labels)[1:] # omit 0
    removed_count = 0
    null_points = np.zeros_like(terminal_segs)
    for u in unique_labels:
        temp_seg = np.zeros_like(terminal_segs)
        temp_seg[term_labels == u] = 1
        seg_inds = np.argwhere(term_labels == u)
        seg_length = np.shape(seg_inds)[0]
        if seg_length<seg_thresh:
            endpoint_inds, endpoints = find_endpoints(temp_seg)
            for i in endpoint_inds:
                endpoint_neighborhood = bp_labels[i[0]-1:i[0]+2, i[1]-1:i[1]+2]
                if np.any(endpoint_neighborhood>0):
                    neighborhood = np.zeros_like(terminal_segs)
                    neighborhood[i[0]-1:i[0]+2, i[1]-1:i[1]+2] = 1
                    null_inds = np.argwhere((neighborhood == 1) & (bp_labels>0))[0]
                    null_points[null_inds[0],null_inds[1]] = 1
            null_points[term_labels==u] = 1
            removed_count+=1
            print(str(u) + ' removed due to length')
    new_skel = skel-null_points
    new_skel[new_skel<0] = 0
    new_skel[new_skel>0] = 1
    edges, bp = connect_segments(new_skel)
    return edges, bp, new_skel

def fix_skel_artefacts(skel):
    edges, bp = find_branchpoints(skel)
    edge_count, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    bp_count, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))
    new_edge_num = edge_count+1
    for i in range(1,bp_count):
        connected_segs = find_connected_segments(bp_labels, edge_labels, i)
        print('bp ' + str(i) + '/'+str(bp_count))
        if len(connected_segs)==2:
            print(str(i)+ ' has 2 connections')
            coord_list = []
            bp_conns = np.zeros_like(edges)
            for c in connected_segs:
                bp_conns[edge_labels == c] = c
                temp_seg = np.zeros_like(edges)
                temp_seg[edge_labels == c] = 1
                coords, endpoints = find_endpoints(temp_seg)
                coord_list.append(coords)
            lowest_dist = 500
            for x in coord_list[0]:
                for y in coord_list[1]:
                    this_dist = dist(x,y)
                    if this_dist<lowest_dist:
                        lowest_dist = this_dist
                        end1, end2 = x,y
            bp_labels[bp_labels == i] = 0
            
            rr, cc = line(end1[0],end1[1],end2[0],end2[1])
            for r,c in zip(rr,cc):
                edge_labels[r,c] = new_edge_num #
                
    new_edges = np.zeros_like(edge_labels)
    new_bp = np.zeros_like(edge_labels)
    
    new_edges[edge_labels>0]=1
    new_bp[bp_labels>0]=1
    
    new_skel = new_edges+new_bp
    edge_count, edge_labels = cv2.connectedComponents(new_edges.astype(np.uint8))
    bp_count, bp_labels = cv2.connectedComponents(new_bp.astype(np.uint8))
    
    for i in range(1,edge_count):
        bp_num = branchpoints_per_seg(skel, edge_labels, bp, i)
        temp_seg = np.zeros_like(edges)
        temp_seg[edge_labels == i] =1
        seg_inds = np.argwhere(temp_seg==1)
        seg_length = np.shape(seg_inds)[0]
        if (bp_num <2) and (seg_length<10):
            print('removing seg ' + str(i) + ' of length ' + str(seg_length))
            for x,y in seg_inds:
                edge_labels[x,y] = 0
    
    new_skel = edge_labels+new_bp
    new_skel[new_skel>0] = 1
    new_edges, new_bp = find_branchpoints(new_skel)
    return new_edges, new_bp
    
def skeletonize_vm(label):
    skel = skeletonize(label)
    _,_, skel = prune_terminal_segments(skel)
    edges, bp = fix_skel_artefacts(skel)
    new_skel = edges+bp
    return new_skel, edges, bp

def branchpoints_per_seg(skel, edge_labels, bp, seg_num):
    bp_count, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))
    bp_labels = bp_labels+1
    bp_labels[bp_labels<2]=0
    temp_seg = np.zeros_like(skel)
    temp_seg[edge_labels==seg_num]=1
    temp_seg = temp_seg+bp_labels
    seg_inds = np.argwhere(temp_seg==1)
    seg_lenth = np.shape(seg_inds)[0]
    bp_num = 0
    for i in seg_inds:
        this_tile = temp_seg[i[0]-1:i[0]+2,i[1]-1:i[1]+2]
        unique_bps = np.unique(this_tile)
        unique_bps = np.sum(unique_bps>1)
        bp_num = bp_num + unique_bps
    return bp_num

def find_connected_segments(bp_labels, edge_labels, bp_num):
    this_bp_inds = np.argwhere(bp_labels == bp_num)
    temp_bp = np.zeros_like(bp_labels)
    for i in this_bp_inds:
        temp_bp[i[0],i[1]]=-1
    bp_el = edge_labels+temp_bp
    connected_segs = []
    bp_coords = []
    for x,y in this_bp_inds:
        bp_neighbors = bp_el[x-1:x+2,y-1:y+2]
        if np.any(bp_neighbors>0):
            connections = bp_neighbors[bp_neighbors>0].tolist()
            connected_segs.append(connections)
            for c in connections:    
                bp_coords.append((x,y))
    connected_segs = flatten(connected_segs)
    return connected_segs

#########################################################
# brain specific functions
 
def brain_seg(im, filter = 'meijering', sigmas = range(1,8,1), hole_size = 50, ditzle_size = 500, thresh = 60, preprocess = True):
    if preprocess == True:
        im = preprocess_seg(im)
    
    if filter == 'meijering':
        enhanced_im = meijering(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'sato':
        enhanced_im = sato(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'frangi':
        enhanced_im = frangi(im, sigmas = sigmas, mode = 'reflect', black_ridges = False)
    elif filter == 'jerman':
        enhanced_im = jerman(im, sigmas = sigmas, tau = 0.75, brightondark = True, cval=0, mode = 'reflect')
    norm = np.round(enhanced_im/np.max(enhanced_im)*255).astype(np.uint8)
    
    enhanced_label = np.zeros_like(norm)
    enhanced_label[norm>thresh] =1
    
    
    kernel = np.ones((6,6),np.uint8)
    label = cv2.morphologyEx(enhanced_label.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    _, label = fill_holes(label.astype(np.uint8),hole_size)
    label = remove_small_objects(label,ditzle_size)
    
    return label

def crop_brain_im(im,label = None):
    new_im = im[300:750,50:500]
    if label is None:
        return new_im
    else:
        new_label = label[300:750,50:500]
        return new_im, new_label

def jerman(im, sigmas = range(1,10,2), tau = 0.75, brightondark = True, cval=0, mode = 'reflect'):
    if brightondark == False:
        im = invert(im)
    vesselness = np.zeros_like(im)
    for i,sigma in enumerate(sigmas):
        lambda1, lambda2 = compute_hessian_eigenvalues(im, sigma, sorting='abs', mode=mode, cval=cval)
        if brightondark == True:
            lambda2 = -lambda2
        lambda3 = lambda2
        
        lambda_rho = lambda3
        lambda_rho = np.where((lambda3 >0) & (lambda3<= tau*np.max(lambda3)), tau*np.max(lambda3), lambda_rho)
        
        lambda_rho[lambda3<0]=0
        
        response = np.zeros_like(lambda1)
        response = lambda2*lambda2*(lambda_rho-lambda2)*27/np.power(lambda2+lambda_rho,3)
        
        response = np.where((lambda2>=lambda_rho/2) & (lambda_rho>0),1,response)
        response = np.where((lambda2<=0) | (lambda_rho<=0),0,response)
        
        if i == 0:
            vesselness = response
        else:
            vesselness = np.maximum(vesselness, response)
    vesselness = vesselness/np.max(vesselness)
    vesselness[vesselness<0.001]=0
    return vesselness
#####################################################################
# Diameter

def whole_anatomy_diameter(im, seg, edge_labels, minimum_length = 25, pad_size = 50):
    unique_edges = np.unique(edge_labels)
    unique_edges = np.delete(unique_edges,0)
    
    edge_label_pad = np.pad(edge_labels,pad_size)
    seg_pad = np.pad(seg, pad_size)
    im_pad = np.pad(im,pad_size)
    full_viz = np.zeros_like(seg_pad)
    diameters = []
    for i in unique_edges:
        seg_length = len(np.argwhere(edge_label_pad == i))
        if seg_length>minimum_length:
            _, temp_diam, temp_viz = visualize_vessel_diameter(edge_label_pad, i, seg_pad, im_pad)
            diameters.append(temp_diam)
            full_viz = full_viz + temp_viz
    im_shape = edge_label_pad.shape
    full_viz_no_pad = full_viz[pad_size:im_shape[0]-pad_size,pad_size:im_shape[1]-pad_size]
    
    return full_viz_no_pad, diameters

def visualize_vessel_diameter(edge_labels, segment_number, seg, im, use_label = False):
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

#####################################################################
# DEPRECATED

def segment_vessels(image,k = 12, hole_size = 500, ditzle_size = 750, bin_thresh = 2):
    image = cv2.medianBlur(image.astype(np.uint8),7)
    image, background = subtract_background_rolling_ball(image, 400, light_background=False,
                                         use_paraboloid=False, do_presmooth=True)
    im_vector = image.reshape((-1,)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(im_vector,k,None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    center = center.astype(np.uint8)
    label_im = label.reshape((image.shape))
    seg_im = np.zeros_like(label_im)
    for i in np.unique(label_im):
        seg_im = np.where(label_im == i, center[i],seg_im)
    
    _, seg_im = cv2.threshold(seg_im.astype(np.uint16), bin_thresh, 255, cv2.THRESH_BINARY)
    
    _, seg_im = fill_holes(seg_im.astype(np.uint8),hole_size)
    seg_im = remove_small_objects(seg_im,ditzle_size)
    
    return seg_im

