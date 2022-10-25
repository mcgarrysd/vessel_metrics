#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:42:00 2022

manual param validation

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
from scipy.stats import ttest_ind

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/SE4_manual_params/'
data_files = os.listdir(data_path)

segments_label = []
segments_seg = []

edges_unfixed = []
bp_unfixed = []

bp_label = []
bp_seg = []
for file in data_files:
    print(file)
    im = cv2.imread(data_path+file+'/img_preproc.png',0)
    seg = vm.brain_seg(im, filter = 'meijering', thresh = 40, preprocess = True)
    
    skel_raw = skeletonize(seg)
    edges_raw, bp_raw = vm.find_branchpoints(skel_raw)
    edge_count_raw,_ = cv2.connectedComponents(edges_raw)
    bp_count_raw,_ = cv2.connectedComponents(bp_raw)
    edges_unfixed.append(edge_count_raw)
    bp_unfixed.append(bp_count_raw)
            
    skel, edges, bp = vm.skeletonize_vm(seg)
    seg_count, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    bp_count, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))
    
    segments_seg.append(seg_count)
    bp_seg.append(bp_count)
    
    label = cv2.imread(data_path+file+'/label.png',0)
    label[label>0] = 1
    skel2, edges2, bp2 = vm.skeletonize_vm(label)
    seg_count2, _ =  cv2.connectedComponents(edges2.astype(np.uint8))
    bp_count2, _ = cv2.connectedComponents(bp2.astype(np.uint8))
    
    segments_label.append(seg_count2)
    bp_label.append(bp_count2)

manual_seg = [40,35,29,29]
manual_bp = [25,23,22,18]

hand_drawn_seg = [50,41,33,41]
hand_drawn_bp = [26,21,13,19]

pd_hd_seg = np.mean(percent_dif(manual_seg, hand_drawn_seg))
pd_hd_bp = np.mean(percent_dif(manual_bp, hand_drawn_bp))

pd_ec_unf = percent_dif(manual_seg, edges_unfixed)
pd_bp_unf = percent_dif(manual_bp, bp_unfixed)
pd_ec_fixed = percent_dif(manual_seg, segments_seg)
pd_bp_fixed = percent_dif(manual_bp, bp_seg)

print('segment unf: ' + str(np.mean(pd_ec_unf)))
print('segment fixed: ' + str(np.mean(pd_ec_fixed)))

print('bp unf: ' + str(np.mean(pd_bp_unf)))
print('bp fixed: ' + str(np.mean(pd_bp_fixed)))


def percent_dif(ground_truth, label):
    perc_dif = []
    for i,j in zip(ground_truth, label):
        dif = i - j
        perc_dif.append(dif/i)
    perc_dif = [abs(i) for i in perc_dif]
    return perc_dif