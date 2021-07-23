#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:46:39 2021

Generates annotation samples to test segmentation algorithm on

@author: sean
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import vessel_metrics as vm
from czifile import CziFile

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/'
output_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/annotation_samples/'
image_files = ['35M-59H inx 48hpf Jul 26 2019 E7 MM.czi', 'flk gata 48hpf Jul 26 2019 E4 no gata.czi', 'flk gata 48hpf Jul 26 2019 E1.czi']

for file in image_files:
    output_name = file.replace(" ","_")
    output_name = output_name.replace('.czi','')
    
    volume = vm.preprocess_czi(data_path, file)
    reslice_volume = vm.reslice_image(volume, 4)
    
    middle_slice = np.floor(np.shape(reslice_volume)[0]/2).astype(np.uint16)
    slice_range = range(middle_slice-1,middle_slice+2)
    for i in slice_range:
        this_slice = reslice_volume[i,:,:]
        file_name = output_name + '_slice' +str(i)+'.png'
        cv2.imwrite(output_path+file_name,this_slice)
        

    
this_file = image_files[0]
slices = range(17,21)
slice_range_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_020221/sliding_window_annot/'

output_name = this_file.replace(" ","_")
output_name = output_name.replace('.czi','')
    
volume = vm.preprocess_czi(data_path, file)
reslice_volume = vm.sliding_window(volume,4)

for i in slices:
    this_slice = reslice_volume[i,:,:]
    file_name = output_name + '_slice' +str(i)+'.png'
    cv2.imwrite(slice_range_path+file_name,this_slice)

    
