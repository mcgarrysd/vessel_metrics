#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:21:01 2022

VM E1 multi scale

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

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/E1_segmentation/'

label_list = []
im_list = []
data_files = os.listdir(data_path)



for file in data_files:
    label_list.append(cv2.imread(data_path+file+'/label.png',0))
    im_list.append(cv2.imread(data_path+file+'/img.png',0))
    
seg_list1 = []
conn_list1 = []
area_list1 = []
length_list1 = []
jacc_list1 = []
Q_list1 = []
for im, label in zip(im_list, label_list):
    seg = vm.brain_seg(im, filter = 'meijering', thresh = 40)
    
    length, area, conn, Q = vm.cal(label.astype(np.uint8), seg.astype(np.uint8))
    conn_list1.append(conn)
    area_list1.append(area)
    length_list1.append(length)
    jacc_list1.append(vm.jaccard(label, seg))
    Q_list1.append(Q)
    seg_list1.append(seg)
    
print('Conn: ' + str(np.mean(conn_list1))+' Area: ' + str(np.mean(area_list1))+' Length: ' + str(np.mean(length_list1))+' Jacc: ' + str(np.mean(jacc_list1))+' Q: ' + str(np.mean(Q_list1)))

sigma1 = range(1,3,1)
sigma2 = range(3,9,3)
seg_list2 = []
conn_list2 = []
area_list2 = []
length_list2 = []
jacc_list2 = []
Q_list2 = []
for im, label in zip(im_list, label_list):
    seg = vm.multi_scale_seg(im, filter = 'meijering',sigma1 = sigma1, sigma2 = sigma2, thresh = 20)
    
    length, area, conn, Q = vm.cal(label.astype(np.uint8), seg.astype(np.uint8))
    conn_list2.append(conn)
    area_list2.append(area)
    length_list2.append(length)
    jacc_list2.append(vm.jaccard(label, seg))
    Q_list2.append(Q)
    seg_list2.append(seg)


print('Conn: ' + str(np.mean(conn_list2))+' Area: ' + str(np.mean(area_list2))+' Length: ' + str(np.mean(length_list2))+' Jacc: ' + str(np.mean(jacc_list2))+' Q: ' + str(np.mean(Q_list2)))

plt.close('all')
vm.overlay_segmentation(im_list[11],seg_list1[11])
vm.overlay_segmentation(im_list[11],seg_list2[11])