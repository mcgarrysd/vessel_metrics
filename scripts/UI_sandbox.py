#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:14:39 2022

UI sandbox


@author: sean
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import distance # Diameter measurement
import matplotlib.pyplot as plt
import os
from skimage.filters import meijering, hessian, frangi, sato
from skimage.draw import line # just in tortuosity
from bresenham import bresenham # diameter 
from skimage.util import invert
from skimage.filters.ridges import compute_hessian_eigenvalues
import itertools # fixing skeleton
from math import dist
from aicsimageio import AICSImage
from skimage import data, restoration, util # deprecated preproc
import timeit
from skimage.morphology import white_tophat, black_tophat, disk
import easygui
import vessel_metrics as vm
from PIL import Image, ImageFont, ImageDraw

opening_msg = 'Would you like to analyze a single image or batch process a full directory?'
options = ['single image', 'directory', 'cancel']

dir_analysis = easygui.buttonbox(opening_msg, choices = options)

if dir_analysis == 'single image':
    path = easygui.fileopenbox()
elif dir_analysis == 'directory':
    path = easygui.diropenbox()
    
pre_save_msg = 'Would you like to load previously saved settings?'
pre_save = easygui.ynbox(msg = pre_save_msg)

if pre_save == True:
    settings = easygui.fileopenbox()

if pre_save == False:
    title = 'Segmentation settings'
    message = 'Leave blank to use default settings'
    fields = ['Filter','Threshold','sigma1', 'sigma2', 'hole size', 'ditzle size', 'Preprocess? (yes/no)', 'multi scale? (yes/no)']
    segmentation_settings = easygui.multenterbox(message,title, fields)
    settings = make_segmentation_settings(segmentation_settings)

msg = 'Select output directory'
title = 'Output Directory'
output_path = easygui.diropenbox(msg= msg, title = title)

####################################################################
# single image analysis
if dir_analysis == 'single image':
    images_to_analyze = []
    seg_list = []
    dim_list = []
    file_split = path.split('/')[-1]
    file_name = file_split.split('.')[0]
    file_type = file_split.split('.')[-1]
    if file_type == 'czi':
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

        file_names = []
        reslice = vm.reslice_image(img,final_settings[1])
        for i in range(reslice.shape[0]):
            images_to_analyze.append(reslice[i])
            file_names.append(file_name)
            dim_list.append(dims)
            img, dims = vm.preprocess_czi(path,'',channel = final_settings[0])
    
        for s,t in zip(images_to_analyze, file_names):
            seg = segment_with_settings(s,settings)
            seg_list.append(seg)
            this_file = t.split('/')[-1]
            this_slice = t.split('_')[-1]
            suffix = '_'+this_slice+'.png'
            if os.path.exists(output_path+'/'+this_file) == False:
                os.mkdir(output_path+'/'+this_file)
            vessel_labels = make_labelled_image(s, seg)
            seg = seg*200
            out_dir = output_path+'/'+this_file+'/'
            cv2.imwrite(out_dir+'img'+suffix,s)
            cv2.imwrite(out_dir+'label'+suffix,seg)
            cv2.imwrite(out_dir+'vessel_labels'+suffix,vessel_labels)
    else:
        img = cv2.imread(path,0)
        seg = segment_with_settings(img, settings)
    # set up parameter analysis settings
    msg = 'Select which parameters you would like to analyze on your files'
    title = 'Parameter settings'
    choices = ['vessel density', 'branchpoint density', 'network length', 'tortuosity', 'segment length', 'diameter']
    params = easygui.multchoicebox(msg = msg, title = title, choices = choices)
    for s,t,u in zip(images_to_analyze, file_names,seg_list):
        parameter_analysis(s,u,params,output_path,t)
####################################################################
# Directory analysis
 
if dir_analysis == 'directory':
    file_list = os.listdir(path)
    reduced_file_list = []
    file_types = []
    accepted_file_types = ['czi', 'png', 'tif', 'tiff']
    for file in file_list:
        if '.' in file:
            file_type = file.split('.')[-1]
            if file_type in accepted_file_types:
                reduced_file_list.append(file)
                file_types.append(file_type)
    if len(set(file_types)) == 1:
        images_to_analyze = []
        file_names = []
        if file_type == 'czi':
            dim_list = []
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

            for file in file_list:
                img, dims = vm.preprocess_czi(path,'/'+file,channel = final_settings[0])
                
                reslice = vm.reslice_image(img,final_settings[1])
                for i in range(reslice.shape[0]):
                    images_to_analyze.append(reslice[i])
                    file_split = file.split('.')[0]
                    file_names.append(file_split+'_slice'+str(i))
                    dim_list.append(dims)
            for s,t in zip(images_to_analyze, file_names):
                seg = segment_with_settings(s,settings)
                this_file = t.split('_slice')[0]
                this_slice = t.split('_')[-1]
                suffix = '_'+this_slice+'.png'
                if os.path.exists(output_path+'/'+this_file) == False:
                    os.mkdir(output_path+'/'+this_file)
                vessel_labels = make_labelled_image(s, seg)
                seg = seg*200
                out_dir = output_path+'/'+this_file+'/'
                cv2.imwrite(out_dir+'img'+suffix,s)
                cv2.imwrite(out_dir+'label'+suffix,seg)
                cv2.imwrite(out_dir+'vessel_labels'+suffix,vessel_labels)
                
        else:
            #input type is an image rather than a volume
            output_dirs = []
            for file in reduced_file_list:
                images_to_analyze.append(cv2.imread(path+'/'+file,0))
                file_split = file.split('.')[0]
                output_dirs.append(file_split)
            for s,t in zip(images_to_analyze,output_dirs):
                seg = segment_with_settings(s,settings)
                if os.path.exists(output_path+'/'+t) == False:
                    os.mkdir(output_path+'/'+t)
                vessel_labels = make_labelled_image(s, seg)
                seg = seg*200
                cv2.imwrite(output_path+'/'+t+'/img.png',s)
                cv2.imwrite(output_path+'/'+t+'/label.png',seg)
                cv2.imwrite(output_path+'/'+t+'/vessel_labls.png',vessel_labels)
                
    else:
        title = 'multiple file types'
        msg = 'multiple file types detected, consolidate your file types'
        easygui.codebox(msg = msg, title = title, text = msg)




####################################################################
def make_segmentation_settings(settings_list):
    defaults = ['meijering', 40, range(1,8,1), range(10,20,5), 50, 500, True, False]
    final_settings = []
    for s,t in zip(settings_list, defaults):
        if s == '':
            final_settings.append(t)
        else:
            final_settings.append(s)
    settings_dict = {'filter': final_settings[0], 'threshold': final_settings[1], 'sigma1': final_settings[2], 'sigma2': final_settings[3], 'hole size': final_settings[4], 'ditzle size': final_settings[5], 'preprocess': final_settings[6], 'multi scale': final_settings[7]}
    
    settings_dict['filter']=settings_dict['filter'].lower()
    possible_filters = ['meijering', 'jerman', 'frangi', 'sato']
    if settings_dict['filter'] not in possible_filters:
        settings_dict['filter'] = 'meijering'
    
    settings_dict['threshold']=int(settings_dict['threshold'])
    if settings_dict['threshold']<0 or settings_dict['threshold']>255:
        settings_dict['threshold'] = 40
    
    settings_dict['hole size']=int(settings_dict['hole size'])
    if settings_dict['hole size']<=0:
        settings_dict['hole size'] = 40
    
    settings_dict['ditzle size']=int(settings_dict['ditzle size'])
    if settings_dict['ditzle size']<=0: 
        settings_dict['ditzle size'] = 500
        
    if type(settings_dict['sigma1']) == str:
        res = tuple(map(int, settings_dict['sigma1'].split(' ')))
        try:
            new_sigma = range(res[0], res[1], res[2])
        except:
            print('invalid sigma input, sigma is input as start stop step, i.e. 1 8 1 default values will be used')
            settings_dict['sigma1'] = range(1,8,1)
    if type(settings_dict['sigma2']) == str:
        res = tuple(map(int, settings_dict['sigma2'].split(' ')))
        try:
            new_sigma = range(res[0], res[1], res[2])
        except:
            print('invalid sigma input, sigma is input as start stop step, i.e. 1 8 1 default values will be used')
            settings_dict['sigma2'] = range(10,20,5)
    accepted_answers = ['yes', 'no']
    
    if type(settings_dict['preprocess']) == str:
        settings_dict['multi scale']=settings_dict['multi scale'].lower()
        if settings_dict['multi scale'] not in accepted_answers:
            settings_dict['multi scale'] = False
        if settings_dict['multi scale'] == 'yes':
            settings_dict['multi scale'] = True
        if settings_dict['multi scale'] == 'no':
            settings_dict['multi scale'] = False
        
    if type(settings_dict['preprocess']) == str:
        settings_dict['preprocess']=settings_dict['preprocess'].lower()
        accepted_answers = ['yes', 'no']
        if settings_dict['preprocess'] not in accepted_answers:
            settings_dict['preprocess'] = True
        if settings_dict['preprocess'] == 'yes':
            settings_dict['preprocess'] = True
        if settings_dict['preprocess'] == 'no':
            settings_dict['preprocess'] = False
    return settings_dict
        

def segment_with_settings(im, settings):
    seg = vm.segment_image(im, filter = settings['filter'], sigma1 = settings['sigma1'], sigma2 = settings['sigma2'], hole_size = settings['hole size'], ditzle_size = settings['ditzle size'], preprocess = settings['preprocess'], multi_scale = settings['multi scale'])
    return seg

def parameter_analysis(im, seg, params,output_path, file_name, slice_num = ''):
    seg[seg>0] = 1
    skel, edges, bp = skeletonize_vm(seg)
    edge_count, edge_labels = cv2.connectedComponents(edges)
    overlay = seg*150+skel*200
    cv2.imwrite(output_path+'/'+file_name+'/vessel_centerlines'+slice_num+'.png',seg)
    segment_count = list(range(0, edge_count))
    if 'vessel density' in params:
        density_image, density_array, overlay = vessel_density(im, seg, 16,16)
        vm.overlay_segmentation(im, density_image)
        cv2.imwrite(output_path+'/'+t+'/label_'+slice_num+'.png',seg)
        plt.savefig(output_path+'/'+file_name+'/vessel_density_'+slice_num+'.png', bbox_inches = 'tight')
        plt.close('all')
        cv2.imwrite(output_path+'/'+file_name+'/vessel_density_overlay_'+slice_num+'.png', overlay)
        out_dens = list(zip(list(range(0,255)), density_array))
        np.savetxt(output_path+'/'+file_name+'/vessel_density_'+slice_num+'.txt', out_dens, fmt = '%.1f')
    if 'branchpoint density' in params:
        bp_density, overlay = branchpoint_density(seg)
        np.savetxt(output_path+'/'+file_name+'/network_length'+slice_num+'.txt', net_length_out, fmt = '%.1f')
    if 'network length' in params:
        net_length = vm.network_length(edges)
        net_length_out = []
        net_length_out.append(net_length)
        np.savetxt(output_path+'/'+file_name+'/network_length'+slice_num+'.txt', net_length_out, fmt = '%.1f')
    if 'tortuosity' in params:
        tort_output = tortuosity(edge_labels)
        np.savetxt(output_path+'/'+file_name+'/tortuosity_'+slice_num+'.txt', tort_output, fmt = '%.1f')
    if 'segment length' in params:
        _, length = vm.vessel_length(edge_labels)
        out_length = zip(segment_count,length)
        np.savetxt(output_path+'/'+file_name+'/vessel_length_'+slice_num+'.txt', out_length, fmt = '%.1f')
    if 'diameter' in params:
        viz, diameters = whole_anatomy_diameter(im, seg, edge_labels, minimum_length = 25, pad_size = 50)
        np.savetxt(output_path+'/'+file_name+'/vessel_density_'+slice_num+'.txt', diameters, fmt = '%.1f')
    
    return

def make_labelled_image(im, seg):
    skel, edges, bp = vm.skeletonize_vm(seg)
    _, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    unique_labels = np.unique(edge_labels)[1:]
    overlay = seg*50+edges*200
    output = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for i in unique_labels:
        print(i)
        seg_temp = np.zeros_like(im)
        seg_temp[edge_labels == i] = 1
        if np.sum(seg_temp)>5:
            midpoint = vm.segment_midpoint(seg_temp)
            text_placement = [midpoint[1], midpoint[0]]
            output = cv2.putText(img = output, text = str(i), org=text_placement, fontFace = 3, fontScale = 1, color = (0,255,255))
    
    return output

def vessel_density(im,label, num_tiles_x, num_tiles_y):
    density = np.zeros_like(im).astype(np.float16)
    density_array = []
    label[label>0] = 1
    step_x = np.round(im.shape[0]/num_tiles_x).astype(np.int16)
    step_y = np.round(im.shape[1]/num_tiles_y).astype(np.int16)
    overlay = np.copy(im)
    overlay = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    count = 0 
    for x in range(0,im.shape[0], step_x):
        for y in range(0,im.shape[1], step_y):
            count+=1
            tile = label[x:x+step_x-1,y:y+step_y-1]
            numel = tile.shape[0]*tile.shape[1]
            tile_density = np.sum(tile)/numel
            tile_val = np.round(tile_density*1000)
            density[x:x+step_x-1,y:y+step_y-1] = tile_val
            density_array.append(tile_val)
            
            text_placement = [np.round((x+x+step_x-1)/2).astype(np.uint), np.round((y+y+step_y-1)/2).astype(np.uint)]
            overlay = cv2.putText(img = overlay, text = str(count), org=text_placement, fontFace = 1, fontScale = 1, color = (0,255,255))
    density = density.astype(np.uint16)
    return density, density_array, overlay


def branchpoint_density(label):
    skel, edges, bp = vm.skeletonize_vm(label)
    _, bp_labels = cv2.connectedComponents(bp, connectivity = 8)
    
    skel_inds = np.argwhere(skel > 0)
    overlay = label*50+edges*100
    output = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bp_density = []
    measurement_number = 0
    for i in range(0,len(skel_inds), 50):
        measurement_number+=1
        x = skel_inds[i][0]; y = skel_inds[i][1]
        this_tile = bp_labels[x-25:x+25,y-25:y+25]
        bp_number = len(np.unique(this_tile))-1
        bp_density.append(bp_number)
        text_placement = [y,x]
        output = cv2.putText(img = output, text = str(measurement_number), org=text_placement, fontFace = 3, fontScale = 1, color = (0,255,255))
    
    
    bp_density = np.array(bp_density)
    bp_density[bp_density<0] = 0
    out_list = list(zip(list(range(1,measurement_number)),bp_density))
    return out_list, output