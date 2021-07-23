#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:51:24 2020

Generates connectivity, area, and length metrics on 
STARE dataset manual annotations

@author: sean
"""

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
import os

data_path = '/home/sean/Documents/Calgary_postdoc/Data/stare/'
img_vk = cv2.imread(data_path + 'labels_vk/' + 'im0001.vk.ppm',0)
img_ah = cv2.imread(data_path + 'labels_ah/' + 'im0001.ah.ppm',0)

def acc_metrics(ground_truth, label):
    ground_truth_binary = np.zeros_like(ground_truth)
    label_binary = np.zeros_like(label)
    
    ground_truth_binary[ground_truth>0] = 1
    label_binary[label>0] = 1
    
    true_pos = np.sum(np.logical_and(ground_truth_binary==1,label_binary==1))
    false_pos = np.sum(np.logical_and(ground_truth_binary==0,label_binary==1))
    true_neg = np.sum(np.logical_and(ground_truth_binary==0,label_binary==0))
    false_neg = np.sum(np.logical_and(ground_truth_binary==1,label_binary==0))
    
    sensitivity = round((true_pos)/(true_pos + false_neg),2)
    specificity = round((true_neg)/(true_neg + false_pos),2)
    accuracy = round((true_neg+ true_pos)/(true_neg+true_pos+false_pos+false_neg),2)
    
    return accuracy, sensitivity, specificity

def cal(ground_truth, label):
    ground_truth_binary = np.zeros_like(ground_truth)
    label_binary = np.zeros_like(label)
    
    ground_truth_binary[ground_truth>0] = 1
    label_binary[label>0] = 1
    
    num_labels_gt, labels_gt = cv2.connectedComponents(ground_truth_binary, connectivity = 8)
    num_labels_l, labels_l = cv2.connectedComponents(label_binary, connectivity = 8)
    
    connectivity = round(1 - np.min([1,np.abs(num_labels_gt-num_labels_l)/np.sum(ground_truth_binary)]),2)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilation_gt = cv2.dilate(ground_truth_binary,kernel)
    dilation_l = cv2.dilate(label_binary,kernel)
    
    dilated_label_union = np.logical_and(ground_truth_binary == 1, dilation_l == 1)
    dilated_gt_union = np.logical_and(label_binary == 1, dilation_gt == 1)
    
    area_numerator = np.sum(np.logical_or(dilated_label_union, dilated_gt_union))
    area_denominator = np.sum(np.logical_or(label_binary,ground_truth_binary))
    
    area = round(area_numerator/area_denominator,2)
    
    gt_skeleton = skeletonize(ground_truth_binary)
    l_skeleton = skeletonize(label_binary)
    
    label_skel_int = np.logical_and(l_skeleton,dilation_gt)
    gt_skel_int = np.logical_and(gt_skeleton,dilation_l)
    
    length_numerator = np.sum(np.logical_or(label_skel_int,gt_skel_int))
    length_denominator = np.sum(np.logical_or(l_skeleton,gt_skeleton))
    
    length = round(length_numerator/length_denominator,2)
    
    return length, area,  connectivity
    
def jaccard(ground_truth, label):
    ground_truth_binary = np.zeros_like(ground_truth)
    label_binary = np.zeros_like(label)
    
    ground_truth_binary[ground_truth>0] = 1
    label_binary[label>0] = 1
    
    intersection = np.sum(np.logical_and(ground_truth_binary, label_binary))
    union = np.sum(np.logical_or(ground_truth_binary, label_binary))
    
    jacc = round(intersection/union,2)
    return jacc

file_list_vk = os.listdir(data_path + 'labels_vk')

plt.close('all')

#for i in range(len(file_list_vk)):
for i in range(1):
    im_id = file_list_vk[i].split(".")[0]
    im_ah_name = im_id + ".ah.ppm"
    im_ex_name = im_id + "-vessels4.ppm"
    
    img_vk = cv2.imread(data_path + 'labels_vk/' + file_list_vk[i],0)
    img_ah = cv2.imread(data_path + 'labels_ah/' + im_ah_name,0)
    img_ex = cv2.imread(data_path + 'labels_experimental/' + im_ex_name,0)
    
    acc_ex, sens_ex, spec_ex = acc_metrics(img_vk, img_ex)
    length_ex, area_ex, conn_ex = cal(img_vk, img_ex)
    jacc_ex = jaccard(img_vk, img_ex)
    
    acc_ah, sens_ah, spec_ah = acc_metrics(img_vk, img_ah)
    length_ah, area_ah, conn_ah = cal(img_vk, img_ah)
    jacc_ah = jaccard(img_vk, img_ah)
    
    ah_string = ('Accuracy: ' + str(acc_ah) + ' Sensitivity: ' + str(sens_ah) + 
    "\n Specificity: " + str(spec_ah) + " Length: " + str(length_ah) + "\n Area: " + str(area_ah) +
    " Connectivity: " + str(conn_ah) + "\n Jaccard: " + str(jacc_ah))
    
    ex_string = ('Accuracy: ' + str(acc_ex) + ' Sensitivity: ' + str(sens_ex) + 
    "\n Specificity: " + str(spec_ex) + " Length: " + str(length_ex) + "\n Area: " + str(area_ex) +
    " Connectivity: " + str(conn_ah) + "\n Jaccard: " + str(jacc_ex))
    
    
    plt.figure(figsize = (10,6));
    plt.subplot(1,3,1);
    plt.imshow(img_vk); plt.title('ground truth')
    plt.subplot(1,3,2);
    plt.imshow(img_ah)
    plt.title(ah_string)
    plt.subplot(1,3,3)
    plt.imshow(img_ex)
    plt.title(ex_string)
    #plt.savefig(data_path + 'accuracy_test/'+ im_id + '.png')
    #plt.close()