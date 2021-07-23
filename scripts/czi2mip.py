#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:18:32 2020

czi2mip

takes folder of czi files and creates single mip images

@author: sean
"""

import os
from czifile import CziFile
import numpy as np
import cv2

input_directory = '/home/sean/Documents/Calgary_postdoc/Data/raw_czi/'
output_directory = '/home/sean/Documents/Calgary_postdoc/Data/czi_projections/'

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
        
        cv2.imwrite(output_directory + output_name_channel, projection)