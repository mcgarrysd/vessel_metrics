#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 08:24:50 2021

deprecated_vm_functions

@author: sean
"""
    
def czi2mip(input_directory):
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
            img_norm = projection/np.max(projection)
            img_adj = np.floor(img_norm*255)        
        cv2.imwrite(output_directory + output_name_channel, projection)


def segment_viewer(segment_number, edge_labels, image):
    segment_inds = np.argwhere(edge_labels == segment_number)
    this_segment = np.zeros_like(edge_labels)
    this_segment[edge_labels == segment_number] = 1
    this_segment = this_segment.astype(np.uint8)
    end_points = find_endpoints(this_segment)
    end_inds = np.argwhere(end_points>0)
    mid_point = np.round(np.mean(end_inds, axis = 0)).astype(np.uint64)
    tile_size = np.abs(np.array([end_inds[0,0]-end_inds[1,0], end_inds[0,1]-end_inds[1,1]])*1.5).astype(np.uint8)
    tile = image[mid_point[0]-tile_size[0]:mid_point[0]+tile_size[0], mid_point[1]-tile_size[1]:mid_point[1]+tile_size[1]]
    edge_tile = edge_labels[mid_point[0]-tile_size[0]:mid_point[0]+tile_size[0], mid_point[1]-tile_size[1]:mid_point[1]+tile_size[1]]
    plt.figure(); 
    plt.subplot(1,2,1)
    plt.imshow(tile)
    plt.subplot(1,2,2)
    plt.imshow(edge_tile)
    
def reslice_image(image,thickness):
    num_slices = np.shape(image)[0]
    out_slices = np.ceil(num_slices/thickness).astype(np.uint16)
    output = np.zeros([out_slices, np.shape(image)[1], np.shape(image)[2]])
    count = 0
    for i in range(0,num_slices, thickness):
        if i+thickness<num_slices:
            im_chunk = image[i:i+thickness,:,:]
        else:
            im_chunk = image[i:,:,:]
        
        output[count,:,:] = np.max(im_chunk,axis = 0)
        count+=1
    return output


def crossline_endpoints(label,start,slope):    
    current_point = start
    current_label_val = label[current_point[0],current_point[1]]
    while current_label_val:
        current_x = current_point[0]-1
        current_y = current_point[1]-slope
        current_point = np.int(round(current_x)), np.int(round(current_y))
        current_label_val = label[current_point[0],current_point[1]]
    end_point1 = current_point
    
    current_point = start
    current_label_val = label[current_point[0],current_point[1]]
    while current_label_val:
        current_x = current_point[0]+1
        current_y = current_point[1]+slope
        current_point = np.int(round(current_x)), np.int(round(current_y))
        current_label_val = label[current_point[0],current_point[1]]
    end_point2 = current_point
    return end_point1, end_point2

def find_segment_crossline_length(label,start,slope):
    current_point = start
    current_label_val = label[current_point[0],current_point[1]]
    while current_label_val:
        current_x = current_point[0]-1
        current_y = current_point[1]-slope
        current_point = current_x, current_y
        current_label_val = label[np.int(np.round(current_point[0])),np.int(np.round(current_point[1]))]
    end_point = np.int(np.round(current_x)), np.int(np.round(current_y))
    vessel_radius = distance.chebyshev(end_point,start)
    cross_thickness = vessel_radius*1.5
    return cross_thickness

def distance_along_line(point,slope,distance):
    x_dist = np.sqrt(distance**2/(slope**2+1))
    y_dist = x_dist*slope
    
    x_dist = np.round(x_dist)
    y_dist = np.round(y_dist)
    
    return x_dist, y_dist
    
def calculate_crossline(point, slope, x_dist, y_dist):
    x1 = np.int(point[0]-x_dist)
    y1 = np.int(point[1]-y_dist)
    
    x2 = np.int(point[0]+x_dist)
    y2 = np.int(point[1]+y_dist)
    
    cross_index = list(bresenham(x1,y1,x2,y2))
    
    return cross_index

def vessel_diameter(label, segment):
    segment_endpoints = find_endpoints(segment)
    endpoint_index = np.where(segment_endpoints)
    first_endpoint = endpoint_index[0][0], endpoint_index[1][0]
    segment_indexes = np.argwhere(segment==1)
    
    distances = []
    for i in range(len(segment_indexes[0])):
        this_pt = segment_indexes[0][i], segment_indexes[1][i]
        distances.append(distance.chebyshev(first_endpoint, this_pt))
    sort_indexes = np.argsort(distances)
    sorted_distances = sorted(distances)
    median_index = np.where(sorted_distances == np.median(sorted_distances))[0][0]
    segment_median = sort_indexes[median_index]
    
    start_pt = segment_indexes[sort_indexes[median_index-3]]
    end_pt = segment_indexes[sort_indexes[median_index+3]]
    median_pt = segment_indexes[sort_indexes[median_index]]
    slope = (start_pt[0]-end_pt[0])/(start_pt[1]-end_pt[1])
    cross_slope = -1/slope

    cross_length = find_segment_crossline_length(label, median_pt,cross_slope)
    
    diameters = []
    for i in range(10,len(sort_indexes)-3,10):
        print(i)
        start_pt = segment_indexes[sort_indexes[i-3]]
        end_pt = segment_indexes[sort_indexes[i+3]]
        mid_pt = segment_indexes[sort_indexes[i]]
        slope = (start_pt[0]-end_pt[0])/(start_pt[1]-end_pt[1])

        if slope != 0:
            cross_slope = -1/slope
    
            x_dist, y_dist = distance_along_line(mid_pt, cross_slope, cross_length)
            cross_index = calculate_crossline(mid_pt, cross_slope, x_dist, y_dist)
        else:
            x_dist = 0
            y_dist = cross_length
            
            x1 = np.int(mid_pt[0]-x_dist)
            y1 = np.int(mid_pt[1]-y_dist)
    
            x2 = np.int(mid_pt[0]+x_dist)
            y2 = np.int(mid_pt[1]+y_dist)
    
            cross_index = list(bresenham(x1,y1,x2,y2))
        
        label_intensity = []
        for j in cross_index:
            label[j[0],j[1]] = 3
            label_intensity.append(label[j[0],j[1]])
        this_diameter = np.sum(label_intensity)
        diameters.append(this_diameter)
    return diameters, label, label_intensity
