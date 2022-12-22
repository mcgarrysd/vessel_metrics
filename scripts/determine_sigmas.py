#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:35:13 2022

determine appropriate sigma

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
from aicsimageio import AICSImage

cynthia_test = cv2.imread('/media/sean/SP PHD U3/from_home/UI_test/Aug 19 2022 pdgfrbGFP;kdrlmcherry 32-70hpf DMSO 2/img_slice0.png',0)
mf_test = cv2.imread('/media/sean/SP PHD U3/from_home/merry_faye_data/nov3/processed/foxf2_homo_13_pericytes/img_slice1.png',0)
murine_test = cv2.imread('/media/sean/SP PHD U3/from_home/murine_data/adam/027.tif',0)

vm.show_im(cynthia_test)
vm.show_im(mf_test)
vm.show_im(murine_test)