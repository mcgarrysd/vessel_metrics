#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:04:16 2023

Vessel metrics user interface

@author: sean
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from aicsimageio import AICSImage
import easygui
import vessel_metrics as vm
from PIL import Image, ImageFont, ImageDraw
import pickle

opening_msg = 'Would you like to analyze a single image or batch process a full directory?'
options = ['single image', 'directory', 'cancel']

dir_analysis = easygui.buttonbox(opening_msg, choices = options)

if dir_analysis == 'single image':
    path = easygui.fileopenbox()
elif dir_analysis == 'directory':
    path = easygui.diropenbox()
    
msg = 'Select output directory'
title = 'Output Directory'
output_path = easygui.diropenbox(msg= msg, title = title)

pre_save_msg = 'Would you like to load previously saved settings?'
pre_save = easygui.ynbox(msg = pre_save_msg)

if pre_save == True:
    settings_path = easygui.fileopenbox()
    all_settings = vm.load_settings(settings_path)
    settings = all_settings[0]
    params = all_settings[1]
    if len(all_settings)==3:
        final_settings = all_settings[2]
    save_ans = False

if pre_save == False:
    settings_exp_msg = 'Click yes if you would like to see an explanation of the input parameters for your segmentation'
    settings_exp = easygui.ynbox(msg = settings_exp_msg)
    
    if settings_exp == True:
        filters = 'Filter: Choice of vessel enhancement filters. meijering, sato, frangi, or jerman.\n Meijering default, jerman recommended for dim images.\n'
        Thresh = 'Threshold: Threshold between 0 and 255, default 40.\n Vessel metrics performs a contrast stretch prior to thresholding.\n'
        Sigmas = 'Sigma: Sigma parameters are input to the vessel enhancement filter, determines the size of vessels the filters will enhance.\n Set these values for an even range up to 80% of your max vessel size. i.e. for vessels betwene 1-10 pixels in diameter (1,8,1) is a good choice of sigma.\n Sigma 2 is used in cases where large discrepancies exist in the size of the vessels.\n Set sigma 1 for small vessels and sigma 2 for large vessels.\n'
        Hole = 'Hole size: Hole size describes the area in pixels of holes in solid structures that are automatically filled in in post processing\n' 
        Ditzle = 'Ditzle size: Ditzle size removes binary objects in the final segmentation smaller than this threshold size.\n'
        Preprocess = 'Preprocess: Whether to apply vessel metrics preprocessing\n'
        Multi_scale = 'Multi scale: Whether to include the sigma2 parameter in calculations\n'
        msg = 'Refer to github.com/mcgarrysd/vessel_metrics for readme file'
        easygui.codebox(title = 'Parameter explanation', msg = msg, text =  [filters, Thresh, Sigmas, Hole, Ditzle, Preprocess, Multi_scale])
    
    title = 'Segmentation settings'
    message = 'Leave blank to use default settings'
    fields = ['Filter','Threshold','sigma1', 'sigma2', 'hole size', 'ditzle size', 'Preprocess? (yes/no)', 'multi scale? (yes/no)']
    segmentation_settings = easygui.multenterbox(message,title, fields)
    settings = vm.make_segmentation_settings(segmentation_settings)
    msg = 'Select which parameters you would like to analyze on your files'
    title = 'Parameter settings'
    choices = ['vessel density', 'branchpoint density', 'network length', 'tortuosity', 'segment length', 'diameter']
    params = easygui.multchoicebox(msg = msg, title = title, choices = choices)
    save_settings_msg = 'Would you like to save these settings for future experiments? Settings are saved as settings.data in the directory your image file is located in.'
    save_ans = easygui.ynbox(msg = save_settings_msg)
    

images_to_analyze = []
file_names = []
dim_list = []
seg_list = []


####################################################################
# single image analysis
if dir_analysis == 'single image':
    temp_path = path.replace('\\','/') # fixes issues with windows paths
    file_split = temp_path.split('/')[-1]
    file_name = file_split.split('.')[0]
    file_type = file_split.split('.')[-1]
    print('file name: ' + file_name + '\n file type: ' + file_type)
    # set up parameter analysis settings
    if file_type == 'czi':
        if pre_save == False:
            title = 'CZI detected'
            message = 'How would you like your file processed? Blank fields will autopopulate with 0, 20'
            fields = ['channel (integer beginning with 0)','number of slices per projection']
            czi_settings = easygui.multenterbox(message, title, fields)
            default_settings = [0,20]        
            final_settings = []
            for s,t in zip(czi_settings, default_settings):
                if s == '':
                    final_settings.append(t)
                else:
                    final_settings.append(s)
            final_settings[0] = int(final_settings[0])
            final_settings[1] = int(final_settings[1])
        
        img, dims = vm.preprocess_czi(path,'',channel = final_settings[0])
            
        # check for silliness
        if final_settings[0] > img.shape[1]:
            final_settings[0] = default_settings[0]
        if final_settings[1] > img.shape[2]:
            final_settings[1] = default_settings[1]

        all_settings = [settings, params, final_settings]
        
        file_names = []
        reslice = vm.reslice_image(img,final_settings[1])
        for i in range(reslice.shape[0]):
            this_file = file_name+'_slice'+str(i)
            images_to_analyze.append(reslice[i])
            file_names.append(this_file)
            dim_list.append(dims)
    
        seg_list = vm.analyze_images(images_to_analyze, file_names, settings, output_path)
    else:
        img = cv2.imread(path,0)
        seg = vm.segment_with_settings(img, settings)
        all_settings = [settings, params]
    for s,t,u in zip(images_to_analyze, file_names, seg_list):
        vm.parameter_analysis(s,u,params,output_path,t)

##################################################################
# Directory analysis

if dir_analysis == 'directory':
    file_list = os.listdir(path)
    reduced_file_list, file_types = vm.generate_file_list(file_list)
    if len(set(file_types)) == 1:
        images_to_analyze = []
        file_names = []
        if file_types[0] == 'czi':
            dim_list = []
            if pre_save == False:
                title = 'CZI detected'
                message = 'How would you like your file processed? Blank fields will autopopulate with 0, 20'
                fields = ['channel (integer beginning with 0)','number of slices per projection']
                czi_settings = easygui.multenterbox(message, title, fields)
                default_settings = [0,20]
                final_settings = []
                for s,t in zip(czi_settings, default_settings):
                    if s == '':
                        final_settings.append(t)
                    else:
                        final_settings.append(s)
                final_settings[0] = int(final_settings[0])
                final_settings[1] = int(final_settings[1])
                all_settings = [settings, params, final_settings]
            for file in reduced_file_list:
                img, dims = vm.preprocess_czi(os.path.join(path,file),'',channel = final_settings[0])
                
                reslice = vm.reslice_image(img,final_settings[1])
                for i in range(reslice.shape[0]):
                    images_to_analyze.append(reslice[i])
                    file_split = file.split('.')[0]
                    file_names.append(file_split+'_slice'+str(i))
                    dim_list.append(dims)
            seg_list = vm.analyze_images(images_to_analyze, file_names, settings, output_path)
                
        else:
            #input type is an image rather than a volume
            output_dirs = []
            all_settings = [settings, params]
            for file in reduced_file_list:
                images_to_analyze.append(cv2.imread(os.path.join(path,file),0))
                file_split = file.split('.')[0]
                output_dirs.append(file_split)
            seg_list = vm.analyze_images(images_to_analyze, file_names, settings, output_path)
        for s,t,u in zip(images_to_analyze, file_names, seg_list):
            vm.parameter_analysis(s,u,params,output_path,t)
                
    else:
        title = 'multiple file types'
        msg = 'multiple file types detected, consolidate your file types'
        easygui.codebox(msg = msg, title = title, text = msg)        

if save_ans == True:
    settings_path = os.path.join(output_path,'settings.data')
    vm.save_settings(all_settings, settings_path)
