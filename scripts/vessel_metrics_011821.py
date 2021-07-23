#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 08:08:43 2021

@author: sean
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skan import draw
from skan import skeleton_to_csgraph
from scipy.spatial import distance
import vessel_metrics as vm
from skimage import draw


data_dir = '/home/sean/Documents/Calgary_postdoc/Data/czi_projections/Channel0/contrast_enhanced/E1_good_qual/'

old_label = cv2.imread('/home/sean/Documents/Calgary_postdoc/Data/czi_projections/Channel0/contrast_enhanced/E1_good_qual/label.png',0)
old_label[old_label>0] = 1
old_skel = skeletonize(old_label)

label = cv2.imread(data_dir + 'label.png',0)
image = cv2.imread(data_dir + 'img_enhanced.png',0)

label[label>0] = 1
skel, label = vm.fill_holes(label, 50)

branch_points, edges = vm.find_branchpoints(skel)
num_labels, edge_labels = cv2.connectedComponents(edges, connectivity = 8)
edge_labels, edges = vm.remove_small_segments(edge_labels, 5)

end_points = vm.find_endpoints(edges)

tort, tort_labels = vm.tortuosity(edge_labels,end_points)
segment_number = tort_labels[tort == np.min(tort)]
vm.segment_viewer(segment_number, edge_labels, image)

