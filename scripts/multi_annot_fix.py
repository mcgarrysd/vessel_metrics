#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:34:50 2022

multiple annot - fix annot

@author: sean
"""
data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/multi_annot/idy_annotate/'

files = os.listdir(data_path)

im_list = []
idy_label_list = []
sean_label_list = []
for file in files:
    im_list.append(cv2.imread(data_path+file+'/img.png',0))
    label = cv2.imread(data_path+file+'/label.png',0)
    label[label>0]=1
    sean_label_list.append(label)
    label2 = cv2.imread(data_path+file+'/label_raw.tiff',0)
    label2[label2==255] = 0
    label2[label2>0] = 1
    idy_label_list.append(label2) 
    
i_label_fixed = cv2.resize(idy_label_list[0],(1024,1024))
vm.overlay_segmentation(im_list[0],i_label_fixed+sean_label_list[0]*2)
idy_label_list[0] = i_label_fixed

for i in idy_label_list:
    vm.show_im(i)
    
for i, j in zip(im_list, idy_label_list):
    print("im_dim: "  + str(i.shape) + " label_dim: " + str(j.shape))
    
for im, s_label, i_label in zip(im_list, sean_label_list, idy_label_list):
    vm.overlay_segmentation(im, s_label*2+i_label)
 
    
sean_idy = []
sean_alg = []
idy_alg = []    
for i,j,k in zip(im_list, idy_label_list, sean_label_list):
    seg = vm.brain_seg(i, filter = 'meijering', thresh = 40)
    sean_idy.append(vm.jaccard(j,k))
    sean_alg.append(vm.jaccard(k, seg))
    idy_alg.append(vm.jaccard(j, seg))

######################################################################
# Cynthia 
        
data_path = '/media/sean/SP PHD U3/from_home/vm_manuscript/multi_annot/all_annot/'

# fix and re-save cynthia labels
out_size = (1024,1024)
for file in files:
    im = cv2.imread(data_path+file+'/img.png',0)
    label3_raw = cv2.imread(data_path+file+'/label_ca.png',1)
    label3_crop = label3_raw[205:1605]
    
    label3_resize = cv2.resize(label3_crop, out_size)
    label3_ch1 = label3_resize[:,:,0]
    label3 = np.zeros(out_size)
    label3[(label3_ch1>100) & (label3_ch1<200)] = 1
    label3 = vm.remove_small_objects(label3, 200)
    kernel = np.ones((6,6),np.uint8)
    label3 = cv2.morphologyEx(label3.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(data_path+file+'/label_ca_fixed.png', label3)
    vm.overlay_segmentation(im, label3)


files = os.listdir(data_path)
im_list = []
ca_label_list = []
sm_label_list = []
iv_label_list = []
for file in files:
    im_list.append(cv2.imread(data_path+file+'/img.png',0))
    label = cv2.imread(data_path+file+'/label.png',0)
    label[label>0]=1
    sm_label_list.append(label)
    label2 = cv2.imread(data_path+file+'/label_iv.tiff',0)
    label2[label2==255] = 0
    label2[label2>0] = 1
    iv_label_list.append(label2) 
    label3 = cv2.imread(data_path+file+'/label_ca_fixed.png',0)
    label3[label3>0] = 1
    ca_label_list.append(label3) 
    
overlay = sm_label_list[4]+ca_label_list[4]+iv_label_list[4]
vm.overlay_segmentation(im_list[4],overlay)
vm.overlay_segmentation(im_list[4], sm_label_list[4])
vm.overlay_segmentation(im_list[4], ca_label_list[4])  
vm.overlay_segmentation(im_list[4], iv_label_list[4]) 
seg_disp = vm.brain_seg(im_list[4], filter = 'meijering', thresh = 40)  
vm.overlay_segmentation(im_list[4], seg_disp)     

sean_idy = []
sean_alg = []
idy_alg = []  
sean_cynth = []
idy_cynth = []
cynth_alg = []  
for i,j,k,l in zip(im_list, iv_label_list, sm_label_list, ca_label_list):
    seg = vm.brain_seg(i, filter = 'meijering', thresh = 40)
    sean_idy.append(vm.jaccard(j,k))
    sean_alg.append(vm.jaccard(k, seg))
    idy_alg.append(vm.jaccard(j, seg))
    sean_cynth.append(vm.jaccard(k,l))
    idy_cynth.append(vm.jaccard(j,l))
    cynth_alg.append(vm.jaccard(l,seg))

reliability_matrix = np.zeros((5,6))
reliability_matrix[:,0] = sean_idy
reliability_matrix[:,1] = sean_alg
reliability_matrix[:,2] = sean_cynth
reliability_matrix[:,3] = idy_alg
reliability_matrix[:,4] = idy_cynth
reliability_matrix[:,5] = cynth_alg
