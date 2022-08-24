#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:58:13 2022

Jerman filter

@author: sean
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.util import invert
from skimage.filters.ridges import compute_hessian_eigenvalues
import vessel_metrics as vm

data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/E1_combined/'
data_list = os.listdir(data_path)

file = data_list[13]
im = cv2.imread(data_path+file+'/img.png',0)
im = vm.preprocess_seg(im)
im = vm.contrast_stretch(im)
im_filt = Jerman(im, tau = 0.75, sigmas = range(1,8,1))
plt.figure(); plt.imshow(im_filt)

def Jerman(im, sigmas = range(1,10,2), tau = 0.75, brightondark = True, cval=0, mode = 'reflect'):
    if brightondark == False:
        im = invert(im)
    vesselness = np.zeros_like(im)
    for i,sigma in enumerate(sigmas):
        lambda1, lambda2 = compute_hessian_eigenvalues(im, sigma, sorting='abs', mode=mode, cval=cval)
        if brightondark == True:
            lambda2 = -lambda2
        lambda3 = lambda2
        
        lambda_rho = lambda3
        lambda_rho = np.where((lambda3 >0) & (lambda3<= tau*np.max(lambda3)), tau*np.max(lambda3), lambda_rho)
        
        lambda_rho[lambda3<0]=0
        
        response = np.zeros_like(lambda1)
        response = lambda2*lambda2*(lambda_rho-lambda2)*27/np.power(lambda2+lambda_rho,3)
        
        response = np.where((lambda2>=lambda_rho/2) & (lambda_rho>0),1,response)
        response = np.where((lambda2<=0) | (lambda_rho<=0),0,response)
        
        if i == 0:
            vesselness = response
        else:
            vesselness = np.maximum(vesselness, response)
    vesselness = vesselness/np.max(vesselness)
    vesselness[vesselness<0.001]=0
    return vesselness