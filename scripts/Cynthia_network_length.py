#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:27:40 2022

Cynthia_network_length

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
import timeit

data_path = '/media/sean/SP PHD U3/from_home/cynthia_network_length/oct_19/'

file_list = os.listdir(data_path+'Raw/')

t0 = timeit.default_timer()
net_length = []
net_length_um = []
im_name_list = []
mkdir = True
for file in file_list:
    fname = file.split('.')
    fname = fname[0].split(' ')
    und = '_'
    prefix = fname[0]+und+fname[1]+und+fname[-2]+und+fname[-1]
    print(file, prefix)
    
    volume, dims = vm.preprocess_czi(data_path+'Raw/',file, channel = 1)
    slice_range = len(volume)
    slice_thickness = np.round(slice_range/2).astype(np.uint8)
    reslice = vm.reslice_image(volume,slice_thickness)
    if mkdir == True:
        os.mkdir(data_path+'Processed/'+prefix)
    for i in range(reslice.shape[0]):
        this_slice = reslice[i]
        seg = vm.brain_seg(this_slice, thresh = 40)
        
        skel, edges, bp = vm.skeletonize_vm(seg)
        nl = vm.network_length(edges)
        net_length.append(nl)
        net_length_um.append(nl*dims[1])
        overlay = edges*100+seg*50
        
        suffix = '_slice'+str(i)+'.png'
        im_name_list.append(prefix+'_slice'+str(i))

        cv2.imwrite(data_path+'Processed/'+prefix+'/img'+suffix,this_slice)
        cv2.imwrite(data_path+'Processed/'+prefix+'/label'+suffix, overlay)
     
t1 = vm.timer_output(t0)
#file = file_list[0]
#img = AICSImage(data_path+'Raw/'+file)
#
#
#with CziFile(data_path+'Raw/'+file) as czi:
#        image_arrays = czi.asarray()
#        meta = czi.metadata()
#        meta2 = czi.metadata(raw = False)
#        
#root = ET.fromstring(meta)
#
#for child in root:
#    print(child.tag, child.attrib)
    
skel = skeletonize(seg)
_,_, skel = prune_terminal_segments(skel)
new_edges, new_bp = fix_skel_artefacts(skel)

def fix_skel_artefacts(skel):
    edges, bp = find_branchpoints(skel)
    edge_count, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    bp_count, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))
    new_edge_num = edge_count+1
    for i in np.unique(bp_labels[1:]):
        connected_segs = find_connected_segments(bp_labels, edge_labels, i)
        if len(connected_segs)==2:
            coord_list = []
            bp_conns = np.zeros_like(edges)
            for c in connected_segs:
                bp_conns[edge_labels == c] = c
                temp_seg = np.zeros_like(edges)
                temp_seg[edge_labels == c] = 1
                if np.sum(temp_seg == 1):
                    coords = np.argwhere(temp_seg == 1)
                else:
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