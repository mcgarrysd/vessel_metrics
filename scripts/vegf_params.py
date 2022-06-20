#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:33:15 2022

vegf params

@author: sean
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
import glob, os
import pandas as pd
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import meijering, hessian, frangi, sato
from skimage.morphology import skeletonize
from scipy.stats import ttest_ind


generate_data = False
if generate_data:
    data_path = '/media/sean/0012-D687/from_home/vm_manuscript/raw_data/VEGFR Treatment/'
    output_path = '/media/sean/0012-D687/from_home/vm_manuscript/vegf/'
    data_list = os.listdir(data_path)
    count=1
    for file in data_list:
        if 'DMH' in file:
            treatment = 'vegf'
        else:
            treatment = 'dmso'
        volume = vm.preprocess_czi(data_path,file, channel = 0)
        slice_range = len(volume)
        slice_thickness = np.round(slice_range/2).astype(np.uint8)
        reslice = vm.reslice_image(volume,slice_thickness)
        this_slice = reslice[0]
        
        # data_list[2] needs channel 1
        
        im_name = treatment+'_'+str(count)+'.png'
        count+=1
        print(file, im_name)
        
        cv2.imwrite(output_path +im_name, this_slice)

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/vegf/'
data_list = os.listdir(data_path)
img_list = []
time_pt = []
preproc_list = []
seg_list = []
skel_list=[]
bp_list = []
edges_list = []
seg_count_list = []
net_length_list = []
vessel_density_list = []
bpd_list = []
length_list = []
condition_list = []

for file in data_list:
    print(file)
    im = cv2.imread(data_path+file,0)
    img_list.append(im)
    im_preproc = vm.contrast_stretch(im)
    im_preproc = vm.preprocess_seg(im_preproc)
    preproc_list.append(im_preproc)
    seg = vm.brain_seg(im)
    seg_list.append(seg)
    t = file.split('_')
    condition_list.append(t[0])
    skel = skeletonize(seg)
    edges, bp, new_skel = vm.prune_terminal_segments(skel)
    skel_list.append(new_skel)
    edges, bp = vm.connect_segments(new_skel)
    edges_list.append(edges)
    bp_list.append(bp)
    seg_count, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    seg_count_list.append(seg_count)
    net_length = vm.network_length(edges)
    net_length_list.append(net_length)

    _, vessel_density = vm.vessel_density(im, seg, 16, 16)
    vessel_density = np.array(vessel_density)
    vessel_density_mean = np.mean(vessel_density[vessel_density>0])
    vessel_density_list.append(vessel_density_mean)
    
    bp_density = vm.branchpoint_density(new_skel, seg)
    bpd_list.append(np.mean(bp_density[np.nonzero(bp_density)]))
    
    _, length = vm.vessel_length(edge_labels)
    length_list.append(np.mean(length))
    #vm.overlay_segmentation(im, seg)
    
cols = ['condition', 'segment count', 'network length', 'vessel_density', 'branchpoint density', 'mean length']
df = pd.DataFrame(list(zip(condition_list,seg_count_list,net_length_list,vessel_density_list, bpd_list, length_list)),columns = cols) 
df.boxplot(column = 'network length', by = 'condition')
df.boxplot(column = 'segment count', by = 'condition')
df.boxplot(column = 'vessel_density', by = 'condition')
df.boxplot(column = 'branchpoint density', by = 'condition')
df.boxplot(column = 'mean length', by = 'condition')


inds_c = np.where(df['condition']=='vegf')
inds_d = np.where(df['condition']=='dmso')

_,p_val_seg_count = ttest_ind(df['segment count'][inds_c[0]],df['segment count'][inds_d[0]])

stat,p_val_net_length = ttest_ind(df['network length'][inds_c[0]],df['network length'][inds_d[0]])

_,p_val_vd = ttest_ind(df['vessel_density'][inds_c[0]],df['vessel_density'][inds_d[0]])

_,p_val_bpd = ttest_ind(df['branchpoint density'][inds_c[0]],df['branchpoint density'][inds_d[0]])

_,p_val_length = ttest_ind(df['mean length'][inds_c[0]],df['mean length'][inds_d[0]])


file = data_list[3]
