#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:19:48 2021

@author: sean
"""

import cv2
import numpy as np
import vessel_metrics as vm
from czifile import CziFile

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/'
data_files = ['35M-59H inx 48hpf Apr 14 2019 E2.czi', '35M-59H inx 48hpf Apr 14 2019 E9.czi',\
'flk gata 48hpf Jul 26 2019 E5.czi', 'flk gata inx 48hpf Apr 14 2019 E4.czi']

vol = vm.preprocess_czi(data_path, data_files[0])
vol = vm.sliding_window(vol, 4)

im = vol[15,:,:]
save_im_flag = True
if save_im_flag:
    cv2.imwrite('/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/sample_images/boudegga/img.png',im)
plt.figure(); plt.imshow(im)
im = im.astype(np.uint16)

label= cv2.imread('/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/sample_images/boudegga/label.png',0)
label[label>0] = 1

seg1 = vm.segment_vessels(im)

clahe_default = cv2.createCLAHE(clipLimit = 40, tileGridSize = (8,8))
cl1 = clahe_default.apply(im)
plt.figure(); plt.imshow(cl1)
cl1_norm = np.round(cl1/np.max(cl1)*255)

clahe_12 = cv2.createCLAHE(clipLimit = 40, tileGridSize = (12,12))
cl2 = clahe_12.apply(im)
cl2_norm = np.round(cl2/np.max(cl2)*255)
plt.figure(); plt.imshow(cl2)

clahe_dif = np.abs(cl1_norm-cl2_norm)
plt.figure(); plt.imshow(clahe_dif)

clahe_16 = cv2.createCLAHE(clipLimit = 40, tileGridSize = (16,16))
cl3 = clahe_16.apply(im)
cl3_norm = np.round(cl3/np.max(cl3)*255)
plt.figure(); plt.imshow(cl3_norm)

clahe_32 = cv2.createCLAHE(clipLimit = 40, tileGridSize = (32,32))
cl4 = clahe_32.apply(im)
cl4_norm = np.round(cl4/np.max(cl4)*255)
plt.figure(); plt.imshow(cl4_norm)

display_dif = False
if display_dif:
    cl_dif = abs(cl1_norm-im)
    plt.figure(); plt.imshow(cl_dif)
    
seg2 = vm.segment_vessels(cl1_norm)
seg3 = vm.segment_vessels(cl2_norm)
seg4 = vm.segment_vessels(cl3_norm)
seg5 = vm.segment_vessels(cl4_norm)

jacc = []
jacc.append(vm.jaccard(label,seg1))
jacc.append(vm.jaccard(label,seg2))
jacc.append(vm.jaccard(label,seg3))
jacc.append(vm.jaccard(label,seg4))
jacc.append(vm.jaccard(label,seg5))

plt.figure(); plt.imshow(seg4)

clahe_20  = cv2.createCLAHE(clipLimit = 20, tileGridSize = (16,16))
cl5 = clahe_20.apply(im)
cl5_norm = np.round(cl5/np.max(cl5)*255)
plt.figure(); plt.imshow(cl5_norm)

clahe_80  = cv2.createCLAHE(clipLimit = 80, tileGridSize = (16,16))
cl6 = clahe_80.apply(im)
cl6_norm = np.round(cl6/np.max(cl6)*255)
plt.figure(); plt.imshow(cl6_norm)

seg6 = vm.segment_vessels(cl5_norm)
seg7 = vm.segment_vessels(cl6_norm)

new_jacc = []
new_jacc.append(vm.jaccard(label,seg4))
new_jacc.append(vm.jaccard(label,seg6))
new_jacc.append(vm.jaccard(label,seg7))

# Comparison where CLAHE is applied
clahe_jpeg = cl3_norm
clahe = cv2.createCLAHE(clipLimit = 40, tileGridSize = (16,16))

raw_czi = vm.preprocess_czi(data_path, data_files[0])
raw_czi = raw_czi.astype(np.uint16)
clahe_czi = np.zeros_like(raw_czi)
for i in range(shape(raw_czi)[0]):
    clahe_czi[i,:,:] = clahe.apply(raw_czi[i,:,:])
clahe_czi = vm.normalize_contrast(clahe_czi)

clahe_vol = vm.sliding_window(clahe_czi,4)
clahe_im = clahe_vol[15,:,:]

clahe_seg = vm.segment_vessels(clahe_im)
clahe_jacc = vm.jaccard(label,clahe_seg)

clahe_overlap = clahe_seg+seg4
label_overlap = seg4+label*2
