#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:27:28 2021

converts list of folders created by labelme into single folder of images and labels

@author: sean
"""

import cv2
import numpy as np
import vessel_metrics as vm
from czifile import CziFile
from copy import deepcopy
import os
import gc
import glob
import shutil

data_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/segmentation/'
label_path = '/home/sean/Documents/Calgary_postdoc/Data/jasper_042821/segmentation/labels/'

json_list = glob.glob(data_path+'*.json')

for json in json_list:
    file_name = json.split('/')[-1]
    root_name = file_name.split('.')[0]
    
    image_name = root_name+'.png'
    shutil.copy(data_path+image_name, data_path+'images/'+image_name)
    shutil.copy(data_path+root_name+'/label.png',data_path+'labels/'+image_name)
