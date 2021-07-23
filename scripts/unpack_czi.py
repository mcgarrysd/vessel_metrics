#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:28:29 2021

unpack czi

@author: sean
"""

input_dir = '/home/sean/Documents/Calgary_postdoc/Data/jasper_0202221/'
output_dir = '/home/sean/Documents/Calgary_postdoc/Data/jasper_0202221/mip/'

import os
from czifile import CziFile
import numpy as np
import cv2
import vessel_metrics as vm

file_list = os.listdir(input_dir)

for i in file_list:
    if '.czi' in i:
        output_name = i.replace(" ","_")
        output_name = output_name.replace('.czi','')
        with CziFile(input_dir + i) as czi:
            image_arrays = czi.asarray()
    
        image = np.squeeze(image_arrays)
        
        if image.ndim==3:
            image = image[np.newaxis,:,:,:]
        
        output_name_channel = output_name + '.png'
        norm = vm.normalize_contrast(image[0,:,:,:])
        projection = np.max(norm, axis = 0)
            
        cv2.imwrite(output_dir + output_name_channel, projection)
        print('finished image ' + i)
    else:
        print('czi not found in ' + i)