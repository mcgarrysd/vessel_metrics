#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 07:51:19 2021

Segment tools - tools for extracting metrics from a binary mask
of blood vessel image

@author: sean
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import distance  # Diameter measurement
import matplotlib.pyplot as plt
import os
from skimage.filters import meijering, hessian, frangi, sato
from skimage.draw import line  # just in tortuosity
from bresenham import bresenham  # diameter
from skimage.util import invert
from skimage.filters.ridges import hessian_matrix_eigvals
from skimage.feature import hessian_matrix
import itertools  # fixing skeleton
from math import dist
from aicsimageio import AICSImage
import aicsimageio
from skimage import data, restoration, util  # deprecated preproc
import timeit
from skimage.morphology import white_tophat, black_tophat, disk
import pickle
import time
import easygui
from icecream import ic

####################################################################
# functions likely to be useful implementing vessel metrics independently on your own data


def segment_image(
    im,
    im_filter="meijering",
    sigma1=range(1, 8, 1),
    sigma2=range(10, 20, 5),
    hole_size=50,
    ditzle_size=500,
    thresh=60,
    preprocess=True,
    multi_scale=False,
):
    if preprocess == True:
        im = preprocess_seg(im)
    if im_filter == "meijering":
        enh_sig1 = meijering(
            im, sigmas=sigma1, mode="reflect", black_ridges=False
        )
        enh_sig2 = meijering(
            im, sigmas=sigma2, mode="reflect", black_ridges=False
        )
    elif im_filter == "sato":
        enh_sig1 = sato(im, sigmas=sigma1, mode="reflect", black_ridges=False)
        enh_sig2 = sato(im, sigmas=sigma2, mode="reflect", black_ridges=False)
    elif im_filter == "frangi":
        enh_sig1 = frangi(
            im, sigmas=sigma1, mode="reflect", black_ridges=False
        )
        enh_sig2 = frangi(
            im, sigmas=sigma2, mode="reflect", black_ridges=False
        )
    elif im_filter == "jerman":
        enh_sig1 = jerman(
            im,
            sigmas=sigma1,
            tau=0.75,
            brightondark=True,
            cval=0,
            mode="reflect",
        )
        enh_sig2 = jerman(
            im,
            sigmas=sigma1,
            tau=0.75,
            brightondark=True,
            cval=0,
            mode="reflect",
        )

    sig1_norm = normalize_contrast(enh_sig1)

    if multi_scale == True:
        sig2_norm = normalize_contrast(enh_sig2)
    else:
        sig2_norm = np.zeros_like(enh_sig2)

    norm = sig1_norm.astype(np.uint16) + sig2_norm.astype(np.uint16)
    enhanced_label = np.zeros_like(norm)
    enhanced_label[norm > thresh] = 1

    kernel = np.ones((6, 6), np.uint8)
    label = cv2.morphologyEx(
        enhanced_label.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )

    _, label = fill_holes(label.astype(np.uint8), hole_size)
    label = remove_small_objects(label, ditzle_size)

    return label


def skeletonize_vm(label):
    skel = skeletonize(label)
    _, _, skel = prune_terminal_segments(skel)
    edges, bp = fix_skel_artefacts(skel)
    new_skel = edges + bp
    return new_skel, edges, bp


def preprocess_seg(
    image,
    radius=50,
    median_size=7,
    upper_lim=255,
    lower_lim=0,
    bright_background=False,
):
    image = normalize_contrast(image)

    image = subtract_background(
        image, radius=radius, light_bg=bright_background
    )
    image = cv2.medianBlur(image.astype(np.uint8), median_size)
    image = contrast_stretch(image, upper_lim=upper_lim, lower_lim=lower_lim)
    return image


def show_im(im):
    plt.figure()
    plt.imshow(im, cmap="gray")


def overlay_segmentation(
    im,
    label,
    alpha=0.5,
    contrast_adjust=False,
    im_cmap="gray",
    label_cmap="jet",
):
    if contrast_adjust:
        im = contrast_stretch(im)
        im = preprocess_seg(im)
    masked = np.ma.masked_where(label == 0, label)
    plt.figure()
    plt.imshow(im, "gray", interpolation="none")
    plt.imshow(masked, "jet", interpolation="none", alpha=alpha)
    plt.show(block=False)


####################################################################


def fill_holes(label_binary, hole_size):
    label_inv = np.bitwise_not(label_binary)
    label_inv[label_inv < 255] = 0
    _, inverted_labels, stats, _ = cv2.connectedComponentsWithStats(label_inv)

    vessel_sizes = stats[:, 4]
    small_vessel_inds = np.argwhere(vessel_sizes < hole_size)

    for v in small_vessel_inds:
        inverted_labels[inverted_labels == v] = 0

    label_mask = np.zeros_like(label_inv)
    label_mask[inverted_labels == 0] = 1

    skel = skeletonize(label_mask)
    return skel, label_mask


def tortuosity(edge_labels):
    edges = np.zeros_like(edge_labels)
    edges[edge_labels > 0] = 1
    coords, end_points = find_endpoints(edges)
    endpoint_labeled = edge_labels * end_points
    unique_labels = np.unique(edge_labels)
    unique_labels = unique_labels[1:]
    tortuosity = []
    labels_used = []
    for u in unique_labels:
        this_segment = np.zeros_like(edge_labels)
        this_segment[edge_labels == u] = 1
        end_inds = np.argwhere(endpoint_labeled == u)
        if end_inds.size == 4:
            end_point_line = line(
                end_inds[0, 0], end_inds[0, 1], end_inds[1, 0], end_inds[1, 1]
            )
            endpoint_distance = np.max(np.shape(end_point_line))
            segment_length = np.sum(this_segment)
            tortuosity.append(endpoint_distance / segment_length)
            labels_used.append(u)
    return tortuosity, labels_used


def remove_small_segments(edge_labels, minimum_length):
    unique_labels = np.unique(edge_labels)
    unique_labels = unique_labels[1:]
    for u in unique_labels:
        this_seg_count = np.shape(np.argwhere(edge_labels == u))[0]
        if this_seg_count < minimum_length:
            edge_labels[edge_labels == u] = 0
    edges = np.zeros_like(edge_labels)
    edges[edge_labels > 0] = 1
    edges = edges.astype(np.uint8)
    _, edge_labels_new = cv2.connectedComponents(edges, connectivity=8)
    return edge_labels_new, edges


def find_branchpoints(skel):
    skel_binary = np.zeros_like(skel)
    skel_binary[skel > 0] = 1
    skel_index = np.argwhere(skel_binary == True)
    tile_sum = []
    neighborhood_image = np.zeros(skel.shape)
    for i, j in skel_index:
        this_tile = skel_binary[i - 1 : i + 2, j - 1 : j + 2]
        tile_sum.append(np.sum(this_tile))
        neighborhood_image[i, j] = np.sum(this_tile)
    branch_points_messy = np.zeros_like(neighborhood_image)
    branch_points_messy[neighborhood_image > 3] = 1
    branch_points_messy = branch_points_messy.astype(np.uint8)

    branch_points = branch_points_messy
    edges = np.zeros_like(branch_points_messy)
    edges = skel.astype(np.uint8) - branch_points_messy

    return edges, branch_points


def find_endpoints(edges):
    edge_binary = np.zeros_like(edges)
    edge_binary[edges > 0] = 1
    edge_index = np.argwhere(edge_binary == True)
    tile_sum = []
    neighborhood_image = np.zeros(edges.shape)

    for i, j in edge_index:
        this_tile = edge_binary[i - 1 : i + 2, j - 1 : j + 2]
        tile_sum.append(np.sum(this_tile))
        neighborhood_image[i, j] = np.sum(this_tile)
    end_points = np.zeros_like(neighborhood_image)
    end_points[neighborhood_image == 2] = 1
    coords = np.argwhere(end_points == 1)
    return coords, end_points


def vessel_length(edge_labels):
    unique_segments, segment_counts = np.unique(
        edge_labels, return_counts=True
    )
    unique_segments = unique_segments[1:]
    segment_counts = segment_counts[1:]
    return unique_segments, segment_counts


def normalize_contrast(image):
    img_norm = image / np.max(image)
    img_adj = np.floor(img_norm * 255)
    return img_adj


def find_terminal_segments(skel, edge_labels):
    skel[skel > 0] = 1
    skel_index = np.argwhere(skel == True)
    tile_sum = []
    neighborhood_image = np.zeros(skel.shape)

    for i, j in skel_index:
        this_tile = skel[i - 1 : i + 2, j - 1 : j + 2]
        tile_sum.append(np.sum(this_tile))
        neighborhood_image[i, j] = np.sum(this_tile)
    terminal_points = np.zeros_like(neighborhood_image)
    terminal_points[neighborhood_image == 2] = 1

    terminal_segments = np.zeros_like(terminal_points)
    unique_labels = np.unique(edge_labels)
    unique_labels = unique_labels[1:]

    for u in unique_labels:
        this_segment = np.zeros_like(terminal_points)
        this_segment[edge_labels == u] = 1
        overlap = this_segment + terminal_points
        if len(np.argwhere(overlap > 1)):
            terminal_segments[edge_labels == u] = 1
    return terminal_segments


def preprocess_czi(input_directory, file_name, channel=0):
    img = AICSImage(
        input_directory + file_name, reader=aicsimageio.readers.CziReader
    )
    dims = img.physical_pixel_sizes
    zdim = dims[0]
    ydim = dims[1]
    xdim = dims[2]
    out_dims = [zdim, ydim, xdim]
    image = np.squeeze(img.data)
    if image.ndim == 4:
        image = image[channel, :, :, :]
    image = normalize_contrast(image)
    return image, out_dims


def czi_projection(volume, axis):
    projection = np.max(volume, axis=axis)
    return projection


def segment_chunk(segment_number, edge_labels, volume):
    segment_inds = np.argwhere(edge_labels == segment_number)
    this_segment = np.zeros_like(edge_labels)
    this_segment[edge_labels == segment_number] = 1
    this_segment = this_segment.astype(np.uint8)
    end_inds, end_points = find_endpoints(this_segment)
    mid_point = np.round(np.mean(end_inds, axis=0)).astype(np.uint64)
    tile_size = np.abs(
        np.array(
            [
                np.shape(volume)[0],
                end_inds[0, 0] - end_inds[1, 0],
                end_inds[0, 1] - end_inds[1, 1],
            ]
        )
        * 1.5
    ).astype(np.uint8)
    chunk = volume[
        :,
        mid_point[0] - tile_size[0] : mid_point[0] + tile_size[0],
        mid_point[1] - tile_size[1] : mid_point[1] + tile_size[1],
    ]
    return chunk


def remove_small_objects(label, size_thresh):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        label.astype(np.uint8)
    )

    object_sizes = stats[:, 4]
    small_obj_inds = np.argwhere(object_sizes < size_thresh)

    for v in small_obj_inds:
        labels[labels == v] = 0

    output = np.zeros_like(labels)
    output[labels > 0] = 1

    return output


def subtract_background(image, radius=50, light_bg=False):
    str_el = disk(radius)
    if light_bg:
        output = black_tophat(image, str_el)
    else:
        output = white_tophat(image, str_el)
    return output


def timer_output(t0):
    t1 = timeit.default_timer()
    elapsed_time = round(t1 - t0, 3)

    print(f"Elapsed time: {elapsed_time}")


def preprocess_seg_deprecated(
    image,
    ball_size=0,
    median_size=7,
    upper_lim=255,
    lower_lim=0,
    bright_background=False,
):
    image = normalize_contrast(image)
    if ball_size == 0:
        ball_size = np.round(image.shape[0] / 3)

    if bright_background == False:
        bg = restoration.rolling_ball(image, radius=ball_size)
        image = image - bg
    else:
        image_inverted = util.invert(image)
        bg_inv = restoration.rolling_ball(image, radius=ball_size)
        image = util.invert(image_inverted - bg_inv)
    image = cv2.medianBlur(image.astype(np.uint8), median_size)
    image = contrast_stretch(image, upper_lim=upper_lim, lower_lim=lower_lim)
    return image


def sliding_window(volume, thickness):
    num_slices = np.shape(volume)[0]
    out_slices = num_slices - thickness
    output = np.zeros([out_slices, np.shape(volume)[1], np.shape(volume)[2]])
    for i in range(0, out_slices):
        im_chunk = volume[i : i + thickness, :, :]
        output[i, :, :] = np.max(im_chunk, axis=0)
    return output


def jaccard(ground_truth, label):
    ground_truth_binary = np.zeros_like(ground_truth)
    label_binary = np.zeros_like(label)

    ground_truth_binary[ground_truth > 0] = 1
    label_binary[label > 0] = 1

    intersection = np.sum(np.logical_and(ground_truth_binary, label_binary))
    union = np.sum(np.logical_or(ground_truth_binary, label_binary))

    jacc = round(intersection / union, 2)
    return jacc


def signal_to_noise(image):
    step_size = [round(z / 8) for z in np.shape(image)]
    mid_point = [round(z / 2) for z in np.shape(image)]

    roi = image[
        mid_point[0] - step_size[0] : mid_point[0] + step_size[0],
        mid_point[1] - step_size[1] : mid_point[1] + step_size[1],
    ]

    px_mean = np.mean(roi)
    px_std = np.std(roi)

    snr = px_mean / px_std

    return snr


def clahe(im, tiles=(16, 16), clip_lim=40):
    cl = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=tiles)
    im = im.astype(np.uint16)
    im = cl.apply(im)
    im = normalize_contrast(im)
    return im


def cal(ground_truth, label):
    ground_truth_binary = np.zeros_like(ground_truth)
    label_binary = np.zeros_like(label)

    ground_truth_binary[ground_truth > 0] = 1
    label_binary[label > 0] = 1

    num_labels_gt, labels_gt = cv2.connectedComponents(
        ground_truth_binary, connectivity=8
    )
    num_labels_l, labels_l = cv2.connectedComponents(
        label_binary, connectivity=8
    )

    connectivity = round(
        1
        - np.min(
            [
                1,
                np.abs(num_labels_gt - num_labels_l)
                / np.sum(ground_truth_binary),
            ]
        ),
        2,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilation_gt = cv2.dilate(ground_truth_binary, kernel)
    dilation_l = cv2.dilate(label_binary, kernel)

    dilated_label_union = np.logical_and(
        ground_truth_binary == 1, dilation_l == 1
    )
    dilated_gt_union = np.logical_and(label_binary == 1, dilation_gt == 1)

    area_numerator = np.sum(
        np.logical_or(dilated_label_union, dilated_gt_union)
    )
    area_denominator = np.sum(np.logical_or(label_binary, ground_truth_binary))

    area = round(area_numerator / area_denominator, 2)

    gt_skeleton = skeletonize(ground_truth_binary)
    l_skeleton = skeletonize(label_binary)

    label_skel_int = np.logical_and(l_skeleton, dilation_gt)
    gt_skel_int = np.logical_and(gt_skeleton, dilation_l)

    length_numerator = np.sum(np.logical_or(label_skel_int, gt_skel_int))
    length_denominator = np.sum(np.logical_or(l_skeleton, gt_skeleton))

    length = round(length_numerator / length_denominator, 2)

    Q = connectivity * length * area

    return length, area, connectivity, Q


def seg_holes(label):
    label_inv = np.zeros_like(label)
    label_inv[label == 0] = 1
    label_inv = label_inv.astype(np.uint8)
    _, labelled_holes, stats, _ = cv2.connectedComponentsWithStats(label_inv)
    return labelled_holes, label_inv, stats


def contrast_stretch(image, upper_lim=255, lower_lim=0):
    c = np.percentile(image, 5)
    d = np.percentile(image, 95)

    stretch = (image - c) * ((upper_lim - lower_lim) / (d - c)) + lower_lim
    stretch[stretch < lower_lim] = lower_lim
    stretch[stretch > upper_lim] = upper_lim

    return stretch


def reslice_image(image, thickness):
    num_slices = np.shape(image)[0]
    out_slices = np.ceil(num_slices / thickness).astype(np.uint16)
    output = np.zeros([out_slices, np.shape(image)[1], np.shape(image)[2]])
    count = 0
    for i in range(0, num_slices, thickness):
        if i + thickness < num_slices:
            im_chunk = image[i : i + thickness, :, :]
        else:
            im_chunk = image[i:, :, :]

        output[count, :, :] = np.max(im_chunk, axis=0)
        count += 1
    return output


def connect_segments(skel):
    skel = np.pad(skel, 50)
    edges, bp = find_branchpoints(skel)
    _, edge_labels = cv2.connectedComponents(edges)

    edge_labels[edge_labels != 0] += 1
    bp_el = edge_labels + bp

    _, bp_labels = cv2.connectedComponents(bp)
    unique_bp = np.unique(bp_labels)
    unique_bp = unique_bp[1:]

    bp_list = []
    bp_connections = []
    new_edges = np.zeros_like(skel)
    new_bp = np.zeros_like(skel)
    for i in unique_bp:
        temp_bp = np.zeros_like(bp_labels)
        temp_bp[bp_labels == i] = 1
        bp_size = np.sum(temp_bp)
        if bp_size > 1:
            this_bp_inds = np.argwhere(temp_bp == 1)

            connected_segs = []
            bp_coords = []
            for x, y in this_bp_inds:
                bp_neighbors = bp_el[x - 1 : x + 2, y - 1 : y + 2]
                if np.any(bp_neighbors > 1):
                    connections = bp_neighbors[bp_neighbors > 1].tolist()
                    connected_segs.append(connections)
                    for c in connections:
                        bp_coords.append((x, y))
            bp_list.append(i)
            bp_connections.append(connected_segs)
            connected_segs = flatten(connected_segs)

            vx = []
            vy = []
            for seg in connected_segs:
                # print('segment ' + str(seg))
                temp_seg = np.zeros_like(bp_labels)
                temp_seg[edge_labels == seg] = 1
                endpoints, endpoint_im = find_endpoints(temp_seg)
                if np.size(endpoints):
                    line = cv2.fitLine(endpoints, cv2.DIST_L2, 0, 0.1, 0.1)
                    vx.append(float(line[0]))
                    vy.append(float(line[1]))

            vx = np.array(vx).flatten().tolist()
            vy = np.array(vy).flatten().tolist()

            v_r = list(zip(np.round(vx, 3), np.round(vy, 3)))
            slope_tolerance = 0.1

            inds = list(range(len(v_r)))
            pair_inds = list(itertools.combinations(inds, 2))
            count = 0
            match = []
            for x, y in itertools.combinations(v_r, 2):
                if np.abs(x[0] - y[0]) < slope_tolerance:
                    if np.abs(x[1] - y[1]) < slope_tolerance:
                        match = pair_inds[count]
                count += 1

            if match:
                c1 = bp_coords[match[0]]
                c2 = bp_coords[match[1]]
                connected_pts = list(bresenham(c1[0], c1[1], c2[0], c2[1]))
                temp_edges = np.zeros_like(bp_labels)
                temp_bp = np.zeros_like(bp_labels)
                for x, y in connected_pts:
                    temp_edges[x, y] = 1
                new_edges = new_edges + temp_edges
                for x, y in this_bp_inds:
                    if temp_edges[x, y] == 0:
                        bp_neighbors = edges[x - 1 : x + 2, y - 1 : y + 2]
                        if np.any(bp_neighbors > 0):
                            temp_bp[x, y] = 1
                new_bp = temp_bp + new_bp
            else:
                new_bp = new_bp + temp_bp
        else:
            new_bp = new_bp + temp_bp
    new_edges = new_edges + edges
    xdim, ydim = np.shape(skel)

    new_edges = new_edges[50 : xdim - 50, 50 : ydim - 50]
    new_bp = new_bp[50 : xdim - 50, 50 : ydim - 50]
    return new_edges, new_bp


def flatten(input_list):
    return [item for sublist in input_list for item in sublist]


def network_length(edges):
    edges[edges > 0] = 1
    net_length = np.sum(edges)
    return net_length


def prune_terminal_segments(skel, seg_thresh=20):
    edges, bp = connect_segments(skel)
    _, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    _, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))

    terminal_segs = find_terminal_segments(skel, edge_labels)
    new_terminal_segs = np.zeros_like(terminal_segs)
    _, term_labels = cv2.connectedComponents(terminal_segs.astype(np.uint8))
    unique_labels = np.unique(term_labels)[1:]  # omit 0
    removed_count = 0
    null_points = np.zeros_like(terminal_segs)
    for u in unique_labels:
        temp_seg = np.zeros_like(terminal_segs)
        temp_seg[term_labels == u] = 1
        seg_inds = np.argwhere(term_labels == u)
        seg_length = np.shape(seg_inds)[0]
        if seg_length < seg_thresh:
            endpoint_inds, endpoints = find_endpoints(temp_seg)
            for i in endpoint_inds:
                endpoint_neighborhood = bp_labels[
                    i[0] - 1 : i[0] + 2, i[1] - 1 : i[1] + 2
                ]
                if np.any(endpoint_neighborhood > 0):
                    neighborhood = np.zeros_like(terminal_segs)
                    neighborhood[i[0] - 1 : i[0] + 2, i[1] - 1 : i[1] + 2] = 1
                    null_inds = np.argwhere(
                        (neighborhood == 1) & (bp_labels > 0)
                    )[0]
                    null_points[null_inds[0], null_inds[1]] = 1
            null_points[term_labels == u] = 1
            removed_count += 1
            # print(str(u) + ' removed due to length')
    new_skel = skel - null_points
    new_skel[new_skel < 0] = 0
    new_skel[new_skel > 0] = 1
    edges, bp = connect_segments(new_skel)
    return edges, bp, new_skel


def fix_skel_artefacts(skel):
    edges, bp = find_branchpoints(skel)
    edge_count, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    bp_count, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))
    new_edge_num = edge_count + 1
    for i in np.unique(bp_labels[1:]):
        connected_segs = find_connected_segments(bp_labels, edge_labels, i)
        if len(connected_segs) == 2:
            coord_list = []
            bp_conns = np.zeros_like(edges)
            for c in connected_segs:
                bp_conns[edge_labels == c] = c
                temp_seg = np.zeros_like(edges)
                temp_seg[edge_labels == c] = 1
                if np.sum(temp_seg == 1):
                    coords = np.argwhere(temp_seg == 1)
                else:
                    coords, endpoints = find_endpoints(temp_seg)
                coord_list.append(coords)
            lowest_dist = 500
            for x in coord_list[0]:
                for y in coord_list[1]:
                    this_dist = dist(x, y)
                    if this_dist < lowest_dist:
                        lowest_dist = this_dist
                        end1, end2 = x, y
            bp_labels[bp_labels == i] = 0

            rr, cc = line(end1[0], end1[1], end2[0], end2[1])
            for r, c in zip(rr, cc):
                edge_labels[r, c] = new_edge_num  #

    new_edges = np.zeros_like(edge_labels)
    new_bp = np.zeros_like(edge_labels)

    new_edges[edge_labels > 0] = 1
    new_bp[bp_labels > 0] = 1

    new_skel = new_edges + new_bp
    edge_count, edge_labels = cv2.connectedComponents(
        new_edges.astype(np.uint8)
    )
    bp_count, bp_labels = cv2.connectedComponents(new_bp.astype(np.uint8))

    for i in range(1, edge_count):
        bp_num = branchpoints_per_seg(skel, edge_labels, bp, i)
        temp_seg = np.zeros_like(edges)
        temp_seg[edge_labels == i] = 1
        seg_inds = np.argwhere(temp_seg == 1)
        seg_length = np.shape(seg_inds)[0]
        if (bp_num < 2) and (seg_length < 10):
            for x, y in seg_inds:
                edge_labels[x, y] = 0

    new_skel = edge_labels + new_bp
    new_skel[new_skel > 0] = 1
    new_edges, new_bp = find_branchpoints(new_skel)
    return new_edges, new_bp


def branchpoints_per_seg(skel, edge_labels, bp, seg_num):
    bp_count, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))
    bp_labels = bp_labels + 1
    bp_labels[bp_labels < 2] = 0
    temp_seg = np.zeros_like(skel)
    temp_seg[edge_labels == seg_num] = 1
    temp_seg = temp_seg + bp_labels
    seg_inds = np.argwhere(temp_seg == 1)
    seg_lenth = np.shape(seg_inds)[0]
    bp_num = 0
    for i in seg_inds:
        this_tile = temp_seg[i[0] - 1 : i[0] + 2, i[1] - 1 : i[1] + 2]
        unique_bps = np.unique(this_tile)
        unique_bps = np.sum(unique_bps > 1)
        bp_num = bp_num + unique_bps
    return bp_num


def find_connected_segments(bp_labels, edge_labels, bp_num):
    this_bp_inds = np.argwhere(bp_labels == bp_num)
    temp_bp = np.zeros_like(bp_labels)
    for i in this_bp_inds:
        temp_bp[i[0], i[1]] = -1
    bp_el = edge_labels + temp_bp
    connected_segs = []
    bp_coords = []
    for x, y in this_bp_inds:
        bp_neighbors = bp_el[x - 1 : x + 2, y - 1 : y + 2]
        if np.any(bp_neighbors > 0):
            connections = bp_neighbors[bp_neighbors > 0].tolist()
            connected_segs.append(connections)
            for c in connections:
                bp_coords.append((x, y))
    connected_segs = flatten(connected_segs)
    return connected_segs


def scatter_boxplot(df, group, column, alpha=0.4):
    grouped = df.groupby(group)
    names, vals, xs = [], [], []
    for i, (name, subdf) in enumerate(grouped):
        names.append(name)
        vals.append(subdf[column].tolist())
        xs.append(np.random.normal(i + 1, 0.04, subdf.shape[0]))
    plt.figure()
    plt.boxplot(vals, labels=names, showfliers=False)
    ngroup = len(vals)
    clevels = np.linspace(0.0, 0.4, ngroup)

    for x, val, clevel in zip(xs, vals, clevels):
        plt.scatter(x, val, c=plt.cm.gray(clevel), alpha=0.4)


#########################################################
# brain specific functions


def brain_seg(
    im,
    filter="meijering",
    sigmas=range(1, 8, 1),
    hole_size=50,
    ditzle_size=500,
    thresh=60,
    preprocess=True,
):
    if preprocess == True:
        im = preprocess_seg(im)

    if filter == "meijering":
        enhanced_im = meijering(
            im, sigmas=sigmas, mode="reflect", black_ridges=False
        )
    elif filter == "sato":
        enhanced_im = sato(
            im, sigmas=sigmas, mode="reflect", black_ridges=False
        )
    elif filter == "frangi":
        enhanced_im = frangi(
            im, sigmas=sigmas, mode="reflect", black_ridges=False
        )
    elif filter == "jerman":
        enhanced_im = jerman(
            im,
            sigmas=sigmas,
            tau=0.75,
            brightondark=True,
            cval=0,
            mode="reflect",
        )
    norm = np.round(enhanced_im / np.max(enhanced_im) * 255).astype(np.uint8)

    enhanced_label = np.zeros_like(norm)
    enhanced_label[norm > thresh] = 1

    kernel = np.ones((6, 6), np.uint8)
    label = cv2.morphologyEx(
        enhanced_label.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )

    _, label = fill_holes(label.astype(np.uint8), hole_size)
    label = remove_small_objects(label, ditzle_size)

    return label


def crop_brain_im(im, label=None):
    new_im = im[300:750, 50:500]
    if label is None:
        return new_im
    else:
        new_label = label[300:750, 50:500]
        return new_im, new_label


def jerman(
    im,
    sigmas=range(1, 10, 2),
    tau=0.75,
    brightondark=True,
    cval=0,
    mode="reflect",
):
    if brightondark == False:
        im = invert(im)
    vesselness = np.zeros_like(im)
    for i, sigma in enumerate(sigmas):
        h_elems = hessian_matrix(im, sigma, mode=mode, cval=cval)
        lambda1, lambda2 = hessian_matrix_eigvals(h_elems)
        if brightondark == True:
            lambda2 = -lambda2
        lambda3 = lambda2

        lambda_rho = lambda3
        lambda_rho = np.where(
            (lambda3 > 0) & (lambda3 <= tau * np.max(lambda3)),
            tau * np.max(lambda3),
            lambda_rho,
        )

        lambda_rho[lambda3 < 0] = 0

        response = np.zeros_like(lambda1)
        response = (
            lambda2
            * lambda2
            * (lambda_rho - lambda2)
            * 27
            / np.power(lambda2 + lambda_rho, 3)
        )

        response = np.where(
            (lambda2 >= lambda_rho / 2) & (lambda_rho > 0), 1, response
        )
        response = np.where((lambda2 <= 0) | (lambda_rho <= 0), 0, response)

        if i == 0:
            vesselness = response
        else:
            vesselness = np.maximum(vesselness, response)
    vesselness = vesselness / np.max(vesselness)
    vesselness[vesselness < 0.001] = 0
    return vesselness


def seg_no_thresh(
    im, filter="meijering", sigmas=range(1, 8, 1), preprocess=True
):
    if preprocess == True:
        im = preprocess_seg(im)

    if filter == "meijering":
        enhanced_im = meijering(
            im, sigmas=sigmas, mode="reflect", black_ridges=False
        )
    elif filter == "sato":
        enhanced_im = sato(
            im, sigmas=sigmas, mode="reflect", black_ridges=False
        )
    elif filter == "frangi":
        enhanced_im = frangi(
            im, sigmas=sigmas, mode="reflect", black_ridges=False
        )
    elif filter == "jerman":
        enhanced_im = jerman(
            im,
            sigmas=sigmas,
            tau=0.75,
            brightondark=True,
            cval=0,
            mode="reflect",
        )
    norm = np.round(enhanced_im / np.max(enhanced_im) * 255).astype(np.uint8)

    return norm


def invert_im(im):
    output = np.zeros_like(im)
    im = vm.normalize_contrast(im)
    output = 255 - im
    return output


def subtract_local_mean(im, size=8, bright_bg=True):
    if bright_bg == True:
        im_pad = np.pad(im, (size, size), "maximum")

    else:
        im_pad = np.pad(im, (size, size), "minimum")

    step = np.floor(size / 2).astype(np.uint8)
    output = np.zeros_like(im_pad).astype(np.float16)
    for x in range(size, im_pad.shape[0] - size):
        for y in range(size, im_pad.shape[1] - size):
            region_mean = np.mean(
                im_pad[x - step : x + step, y - step : y + step]
            ).astype(np.uint8)
            output[x, y] = im_pad[x, y] - region_mean
    output[output < 0] = 0
    output = np.round(output).astype(np.uint8)
    output_no_pad = output[
        size : im_pad.shape[0] - size, size : im_pad.shape[1] - size
    ]
    return output_no_pad


def multi_scale_seg(
    im,
    filter="meijering",
    sigma1=range(1, 8, 1),
    sigma2=range(10, 20, 5),
    hole_size=50,
    ditzle_size=500,
    thresh=40,
    preprocess=True,
):
    if preprocess == True:
        im = preprocess_seg(im)

    if filter == "meijering":
        enh_sig1 = meijering(
            im, sigmas=sigma1, mode="reflect", black_ridges=False
        )
        enh_sig2 = meijering(
            im, sigmas=sigma2, mode="reflect", black_ridges=False
        )
    elif filter == "sato":
        enhanced_im = sato(
            im, sigmas=sigmas, mode="reflect", black_ridges=False
        )
    elif filter == "frangi":
        enh_sig1 = frangi(
            im, sigmas=sigma1, mode="reflect", black_ridges=False
        )
        enh_sig2 = frangi(
            im, sigmas=sigma2, mode="reflect", black_ridges=False
        )
    elif filter == "jerman":
        enh_sig1 = jerman(
            im,
            sigmas=sigma1,
            tau=0.75,
            brightondark=True,
            cval=0,
            mode="reflect",
        )
        enh_sig2 = jerman(
            im,
            sigmas=sigma1,
            tau=0.75,
            brightondark=True,
            cval=0,
            mode="reflect",
        )
    sig1_norm = normalize_contrast(enh_sig1)
    sig2_norm = normalize_contrast(enh_sig2)

    norm = sig1_norm.astype(np.uint16) + sig2_norm.astype(np.uint16)
    enhanced_label = np.zeros_like(norm)
    enhanced_label[norm > thresh] = 1

    kernel = np.ones((6, 6), np.uint8)
    label = cv2.morphologyEx(
        enhanced_label.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )

    _, label = fill_holes(label.astype(np.uint8), hole_size)
    label = remove_small_objects(label, ditzle_size)

    return label


#####################################################################
# Diameter


def whole_anatomy_diameter(
    im, seg, edge_labels, minimum_length=25, pad_size=50
):
    unique_edges = np.unique(edge_labels)
    unique_edges = np.delete(unique_edges, 0)

    edge_label_pad = np.pad(edge_labels, pad_size)
    seg_pad = np.pad(seg, pad_size)
    im_pad = np.pad(im, pad_size)
    full_viz = np.zeros_like(seg_pad)
    diameters = []
    included_segments = []
    for i in unique_edges:
        seg_length = len(np.argwhere(edge_label_pad == i))
        if seg_length > minimum_length:
            included_segments.append(i)
            _, temp_diam, temp_viz = visualize_vessel_diameter(
                edge_label_pad, i, seg_pad, im_pad, pad=False
            )
            diameters.append(temp_diam)
            full_viz = full_viz + temp_viz
    im_shape = edge_label_pad.shape
    full_viz_no_pad = full_viz[
        pad_size : im_shape[0] - pad_size, pad_size : im_shape[1] - pad_size
    ]
    output_diameters = list(zip(included_segments, diameters))
    return full_viz_no_pad, output_diameters


def visualize_vessel_diameter(
    edge_labels, segment_number, seg, im, use_label=False, pad=True
):
    if pad == True:
        pad_size = 25
        edge_labels = np.pad(edge_labels, pad_size)
        seg = np.pad(seg, pad_size)
        im = np.pad(im, pad_size)
    segment = np.zeros_like(edge_labels)
    segment[edge_labels == segment_number] = 1
    segment_median = segment_midpoint(segment)

    vx, vy = tangent_slope(segment, segment_median)
    bx, by = crossline_slope(vx, vy)

    viz = np.zeros_like(seg)
    cross_length = find_crossline_length(bx, by, segment_median, seg)

    if cross_length == 0:
        diameter = 0
        mean_diameter = 0
        print("cross length 0")
        return diameter, mean_diameter, viz

    diameter = []
    segment_inds = np.argwhere(segment)
    for i in range(10, len(segment_inds), 10):
        this_point = segment_inds[i]
        vx, vy = tangent_slope(segment, this_point)
        bx, by = crossline_slope(vx, vy)
        _, cross_index = make_crossline(bx, by, this_point, cross_length)
        if use_label:
            cross_vals = crossline_intensity(cross_index, seg)
            diam = label_diameter(cross_vals)
        else:
            cross_vals = crossline_intensity(cross_index, im)
            diam = fwhm_diameter(cross_vals)
        if diam == 0:
            val = 5
        else:
            val = 10
        for ind in cross_index:
            viz[ind[0], ind[1]] = val
        diameter.append(diam)
    diameter = [x for x in diameter if x != 0]
    if diameter:
        mean_diameter = np.mean(diameter)
    else:
        mean_diameter = 0

    if pad == True:
        im_shape = edge_labels.shape
        viz = viz[
            pad_size : im_shape[0] - pad_size,
            pad_size : im_shape[1] - pad_size,
        ]
    return diameter, mean_diameter, viz


def segment_midpoint(segment):
    endpoint_index, segment_endpoints = find_endpoints(segment)
    segment_indexes = np.argwhere(segment)
    if len(endpoint_index) == 0:
        # vessel has no endpoints, may be a circular segment
        segment_inds = np.argwhere(segment)
        center_of_mass = [
            np.average(segment_indexes[:, 0]),
            np.average(segment_inds[:, 1]),
        ]
        distances = []
        for i in range(len(segment_indexes)):
            this_pt = segment_indexes[i][0], segment_indexes[i][1]
            distances.append(distance.chebyshev(center_of_mass, this_pt))
    else:
        first_endpoint = endpoint_index[0][0], endpoint_index[1][0]
        distances = []
        for i in range(len(segment_indexes)):
            this_pt = segment_indexes[i][0], segment_indexes[i][1]
            distances.append(distance.chebyshev(first_endpoint, this_pt))
    sort_indexes = np.argsort(distances)
    sorted_distances = sorted(distances)
    median_val = np.median(sorted_distances)
    dist_from_median = abs(sorted_distances - median_val)
    median_distance = np.where(dist_from_median == np.min(dist_from_median))[
        0
    ][0]
    segment_median = segment_indexes[median_distance]
    segment_median = segment_median.flatten()
    return segment_median


def tangent_slope(segment, point):
    point = point.flatten()
    crop_im = segment[point[0] - 5 : point[0] + 5, point[1] - 5 : point[1] + 5]
    crop_inds = np.transpose(np.where(crop_im))
    line = cv2.fitLine(crop_inds, cv2.DIST_L2, 0, 0.1, 0.1)
    vx, vy = line[0], line[1]
    return vx, vy


def crossline_slope(vx, vy):
    bx = -vy
    by = vx
    return bx, by


def make_crossline(vx, vy, point, length):
    xlen = vx * length / 2
    ylen = vy * length / 2

    x1 = int(np.round(point[0] - xlen))
    x2 = int(np.round(point[0] + xlen))

    y1 = int(np.round(point[1] - ylen))
    y2 = int(np.round(point[1] + ylen))

    rr, cc = line(x1, y1, x2, y2)
    cross_index = []
    for r, c in zip(rr, cc):
        cross_index.append([r, c])
    coords = x1, x2, y1, y2

    return coords, cross_index


def plot_crossline(im, cross_index, bright=False):
    if bright == True:
        val = 250
    else:
        val = 5
    out = np.zeros_like(im)
    for i in cross_index:
        out[i[0], i[1]] = val
    return out


def find_crossline_length(bx, by, point, im):
    d = 5
    diam = 0
    im_size = im.shape[0]
    while diam == 0:
        d += 5
        coords, cross_index = make_crossline(bx, by, point, d)
        if all(i < im_size for i in coords):
            seg_val = []
            for i in cross_index:
                seg_val.append(im[i[0], i[1]])
            steps = np.where(np.roll(seg_val, 1) != seg_val)[0]
            if steps.size > 0:
                if steps[0] == 0:
                    steps = steps[1:]
                num_steps = len(steps)
                if num_steps == 2:
                    diam = abs(steps[1] - steps[0])
                if num_steps > 2:
                    new_steps = [
                        x - steps[i - 1] if i else None
                        for i, x in enumerate(steps)
                    ][1:]
                    diam = np.max(new_steps)
            if d > 100:
                break
        else:
            break
    length = diam * 2.5
    return length


def crossline_intensity(cross_index, im, plot=False):
    cross_vals = []
    for i in cross_index:
        cross_vals.append(im[i[0], i[1]])
    if plot == True:
        inds = list(range(len(cross_vals)))
        plt.figure()
        plt.plot(inds, cross_vals)
    return cross_vals


def label_diameter(cross_vals):
    steps = np.where(np.roll(cross_vals, 1) != cross_vals)[0]
    if steps.size > 0:
        if steps[0] == 0:
            steps = steps[1:]
        num_steps = len(steps)
        if num_steps == 2:
            diam = abs(steps[1] - steps[0])
        else:
            diam = 0
    else:
        diam = 0
    return diam


def fwhm_diameter(cross_vals):
    peak = np.max(cross_vals)
    half_max = np.round(peak / 2)

    peak_ind = np.where(cross_vals == peak)[0][0]
    before_peak = cross_vals[0:peak_ind]
    after_peak = cross_vals[peak_ind + 1 :]
    try:
        hm_before = np.argmin(np.abs(before_peak - half_max))
        hm_after = np.argmin(np.abs(after_peak - half_max))

        # +2 added because array indexing begins at 0 twice
        diameter = (hm_after + peak_ind) - hm_before + 2
    except:
        diameter = 0
    return diameter


##################################################################
# User interface functions


def make_segmentation_settings(settings_list):
    defaults = [
        "meijering",
        40,
        range(1, 8, 1),
        range(10, 20, 5),
        50,
        500,
        True,
        False,
    ]
    final_settings = []
    for s, t in zip(settings_list, defaults):
        if s == "":
            final_settings.append(t)
        else:
            final_settings.append(s)
    settings_dict = {
        "filter": final_settings[0],
        "threshold": final_settings[1],
        "sigma1": final_settings[2],
        "sigma2": final_settings[3],
        "hole size": final_settings[4],
        "ditzle size": final_settings[5],
        "preprocess": final_settings[6],
        "multi scale": final_settings[7],
    }

    settings_dict["filter"] = settings_dict["filter"].lower()
    possible_filters = ["meijering", "jerman", "frangi", "sato"]
    if settings_dict["filter"] not in possible_filters:
        settings_dict["filter"] = "meijering"

    settings_dict["threshold"] = int(settings_dict["threshold"])
    if settings_dict["threshold"] < 0 or settings_dict["threshold"] > 255:
        settings_dict["threshold"] = 40

    settings_dict["hole size"] = int(settings_dict["hole size"])
    if settings_dict["hole size"] <= 0:
        settings_dict["hole size"] = 40

    settings_dict["ditzle size"] = int(settings_dict["ditzle size"])
    if settings_dict["ditzle size"] <= 0:
        settings_dict["ditzle size"] = 500

    if type(settings_dict["sigma1"]) == str:
        res = tuple(map(int, settings_dict["sigma1"].split(" ")))
        try:
            new_sigma = range(res[0], res[1], res[2])
            settings_dict["sigma1"] = new_sigma
        except:
            print(
                "invalid sigma input, sigma is input as start stop step, i.e. 1 8 1 default values will be used"
            )
            settings_dict["sigma1"] = range(1, 8, 1)
    if type(settings_dict["sigma2"]) == str:
        res = tuple(map(int, settings_dict["sigma2"].split(" ")))
        try:
            new_sigma = range(res[0], res[1], res[2])
            settings_dict["sigma2"] = new_sigma
        except:
            print(
                "invalid sigma input, sigma is input as start stop step, i.e. 1 8 1 default values will be used"
            )
            settings_dict["sigma2"] = range(10, 20, 5)
    accepted_answers = ["yes", "no"]

    if type(settings_dict["preprocess"]) == str:
        settings_dict["multi scale"] = settings_dict["multi scale"].lower()
        if settings_dict["multi scale"] not in accepted_answers:
            settings_dict["multi scale"] = False
        if settings_dict["multi scale"] == "yes":
            settings_dict["multi scale"] = True
        if settings_dict["multi scale"] == "no":
            settings_dict["multi scale"] = False

    if type(settings_dict["preprocess"]) == str:
        settings_dict["preprocess"] = settings_dict["preprocess"].lower()
        accepted_answers = ["yes", "no"]
        if settings_dict["preprocess"] not in accepted_answers:
            settings_dict["preprocess"] = True
        if settings_dict["preprocess"] == "yes":
            settings_dict["preprocess"] = True
        if settings_dict["preprocess"] == "no":
            settings_dict["preprocess"] = False
    return settings_dict


def segment_with_settings(im, settings):
    seg = segment_image(
        im,
        im_filter=settings["filter"],
        sigma1=settings["sigma1"],
        sigma2=settings["sigma2"],
        hole_size=settings["hole size"],
        ditzle_size=settings["ditzle size"],
        preprocess=settings["preprocess"],
        multi_scale=settings["multi scale"],
    )
    return seg


def parameter_analysis(im, seg, params, output_path, file_name):
    seg[seg > 0] = 1
    skel, edges, bp = skeletonize_vm(seg)
    edge_count, edge_labels = cv2.connectedComponents(edges)
    overlay = seg * 150 + skel * 200
    this_file = file_name.split("_slice")[0]
    this_slice = file_name.split("_")[-1]
    if this_file == this_slice:
        this_slice = ""
    cv2.imwrite(
        os.path.join(
            output_path, this_file, "vessel_centerlines" + this_slice + ".png"
        ),
        overlay,
    )
    segment_count = list(range(1, edge_count))
    if "vessel density" in params:
        density_image, density_array, overlay = vessel_density(im, seg, 16, 16)
        overlay_segmentation(im, density_image)
        plt.savefig(
            os.path.join(
                output_path, this_file, "vessel_density_" + this_slice + ".png"
            ),
            bbox_inches="tight",
        )
        plt.close("all")
        cv2.imwrite(
            os.path.join(
                output_path,
                this_file,
                "vessel_density_overlay_" + this_slice + ".png",
            ),
            overlay,
        )
        out_dens = list(zip(list(range(0, 255)), density_array))
        np.savetxt(
            os.path.join(
                output_path, this_file, "vessel_density_" + this_slice + ".txt"
            ),
            out_dens,
            fmt="%.1f",
            delimiter=",",
        )
    if "branchpoint density" in params:
        bp_density, overlay = branchpoint_density(seg)
        np.savetxt(
            os.path.join(
                output_path,
                this_file,
                "branchpoint_density_" + this_slice + ".txt",
            ),
            bp_density,
            fmt="%.1f",
            delimiter=",",
        )
    if "network length" in params:
        net_length = network_length(edges)
        net_length_out = []
        net_length_out.append(net_length)
        np.savetxt(
            os.path.join(
                output_path, this_file, "network_length" + this_slice + ".txt"
            ),
            net_length_out,
            fmt="%.1f",
            delimiter=",",
        )
    if "tortuosity" in params:
        tort_output = tortuosity(edge_labels)
        np.savetxt(
            os.path.join(
                output_path, this_file, "tortuosity_" + this_slice + ".txt"
            ),
            tort_output,
            fmt="%.1f",
            delimiter=",",
        )
    if "segment length" in params:
        _, length = vessel_length(edge_labels)
        out_length = list(zip(segment_count, length))
        np.savetxt(
            os.path.join(
                output_path, this_file, "vessel_length_" + this_slice + ".txt"
            ),
            out_length,
            fmt="%.1f",
            delimiter=",",
        )
    if "diameter" in params:
        viz, diameters = whole_anatomy_diameter(
            im, seg, edge_labels, minimum_length=25, pad_size=50
        )
        np.savetxt(
            os.path.join(
                output_path,
                this_file,
                "vessel_diameter_" + this_slice + ".txt",
            ),
            diameters,
            fmt="%.1f",
            delimiter=",",
        )

    return


#########################################################################
# ROI CODE


def crop_roi(im, roi):
    im_out = im[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
    return im_out


def generate_roi(im):
    im_size = im.shape
    window_size = im_size
    if max(im_size) > 1024:
        aspect_ratio = im_size[0] / im_size[1]
        if aspect_ratio > 1:
            window_size = (1024, int(im_size[1] / aspect_ratio))
        elif aspect_ratio < 1:
            window_size = (1024, int(im_size[1] * aspect_ratio))
        elif aspect_ratio == 1:
            window_size = (1024, 1024)
    else:
        window_size = (im_size[0], im_size[1])
    cv2.namedWindow("Select ROI then space bar", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        "Select ROI then space bar", window_size[0], window_size[1]
    )
    roi = cv2.selectROI("Select ROI then space bar", im.astype(np.uint8))
    cv2.destroyWindow("Select ROI then space bar")
    return roi


def crop_image(im):
    r = generate_roi(im)
    im_out = crop_roi(im, r)
    return im_out, r


def UI_crop_image(im, output_dir, slice_number=""):
    im_out, r = crop_image(im)
    im_copy = im.copy()
    im_roi = cv2.rectangle(
        im_copy, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255, 0, 0), 3
    )
    cv2.imwrite(
        os.path.join(output_dir, "full_im_roi" + slice_number + ".png"), im_roi
    )
    cv2.imwrite(
        os.path.join(output_dir, "im_roi" + slice_number + ".png"), im_out
    )
    return im_out, r


def select_roi(image, output_dir, slice_number=""):
    recrop = True
    while recrop == True:
        cropped_im, r = UI_crop_image(image, output_dir, slice_number)
        choices = ["recrop", "accept crop"]
        msg = "Would you like to recrop this image?"
        img_dirs = os.path.join(output_dir, "im_roi" + slice_number + ".png")
        reply = easygui.buttonbox(msg, choices=choices, image=img_dirs)
        if reply == "accept crop":
            recrop = False
    return cropped_im, r


def parameter_analysis_roi(im, seg, params, output_path, file_name, r):
    seg[seg > 0] = 1
    skel, edges, bp = skeletonize_vm(seg)
    edge_count, edge_labels = cv2.connectedComponents(edges)
    im = crop_roi(im, r)
    seg = crop_roi(seg, r)
    skel = crop_roi(skel, r)
    edges = crop_roi(edges, r)
    edge_labels = crop_roi(edge_labels, r)
    bp = crop_roi(bp, r)
    overlay = seg * 150 + skel * 200
    this_file = file_name.split("_slice")[0]
    this_slice = file_name.split("_")[-1]
    out_text = []
    out_text.append(["ROI_analysis"])
    fname = "ROI_analysis_" + this_slice + ".txt"
    if this_file == this_slice:
        this_slice = ""
        suffix = ".png"
    else:
        suffix = "_" + this_slice + ".png"
    segment_count = list(range(1, edge_count))
    if "vessel density" in params:
        density = vessel_density_roi(im, seg)
        out_text_vd = ["vessel density: %.1f" % (density)]
        out_text.append(out_text_vd)
    if "branchpoint density" in params:
        num_branchpoints, bp_density = branchpoint_density_roi(skel, edges, bp)
        out_text_bpd = [
            "branchpoint density: %.1f" % (bp_density),
            "Num branchpoints: %.1f" % (num_branchpoints),
        ]
        out_text.append(out_text_bpd)
    if "network length" in params:
        net_length = network_length(edges)
        out_text_nl = ["network legnth: %.1f pixels" % (net_length)]
        out_text.append(out_text_nl)
    if "segment length" in params:
        segment_count, length = vessel_length(edge_labels)
        out_text_sl = list(zip(segment_count, length))
        sl_fname = "ROI_analysis_SL" + this_slice + ".txt"
        np.savetxt(
            os.path.join(output_path, this_file, sl_fname),
            out_text_sl,
            fmt="%s",
            delimiter=",",
        )

    new_out = [item for sublist in out_text for item in sublist]
    np.savetxt(
        os.path.join(output_path, this_file, fname),
        new_out,
        fmt="%s",
        delimiter=",",
    )
    return


def vessel_density_roi(im, label):
    density = np.zeros_like(im).astype(np.float16)
    density_array = []
    label[label > 0] = 1
    im_size = im.shape
    density = np.sum(label) / (im_size[0] * im_size[1])
    return density


def branchpoint_density_roi(skel, edges, bp):
    _, bp_labels = cv2.connectedComponents(bp, connectivity=8)
    skel_inds = np.argwhere(skel > 0)
    num_branchpoints = np.max(bp_labels)
    im_size = skel.shape
    bp_density = num_branchpoints / (im_size[0] * im_size[1])
    return num_branchpoints, bp_density


def make_labelled_image(im, seg):
    skel, edges, bp = skeletonize_vm(seg)
    _, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    unique_labels = np.unique(edge_labels)[1:]
    overlay = seg * 50 + edges * 200
    output = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for i in unique_labels:
        seg_temp = np.zeros_like(im)
        seg_temp[edge_labels == i] = 1
        if np.sum(seg_temp) > 5:
            try:
                midpoint = segment_midpoint(seg_temp)
                text_placement = [midpoint[1], midpoint[0]]
                output = cv2.putText(
                    img=output,
                    text=str(i),
                    org=text_placement,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 255),
                )
            except:
                print("segment " + str(i) + " failed")

    return output


def vessel_density(im, label, num_tiles_x, num_tiles_y):
    density = np.zeros_like(im).astype(np.float16)
    density_array = []
    label[label > 0] = 1
    step_x = np.round(im.shape[0] / num_tiles_x).astype(np.int16)
    step_y = np.round(im.shape[1] / num_tiles_y).astype(np.int16)
    overlay = np.copy(im)
    overlay = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    count = 0
    for x in range(0, im.shape[0], step_x):
        for y in range(0, im.shape[1], step_y):
            tile = label[x : x + step_x - 1, y : y + step_y - 1]
            numel = tile.shape[0] * tile.shape[1]
            tile_density = np.sum(tile) / numel
            tile_val = np.round(tile_density * 1000)
            density[x : x + step_x - 1, y : y + step_y - 1] = tile_val
            density_array.append(tile_val)

            text_placement = [
                np.round((y + y + step_y - 1) / 2).astype(np.uint),
                np.round((x + x + step_x - 1) / 2).astype(np.uint),
            ]
            overlay = cv2.putText(
                img=overlay,
                text=str(count),
                org=text_placement,
                fontScale=1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                color=(0, 255, 255),
            )
            count += 1
    density = density.astype(np.uint16)
    return density, density_array, overlay


def branchpoint_density(label):
    skel, edges, bp = skeletonize_vm(label)
    _, bp_labels = cv2.connectedComponents(bp, connectivity=8)

    skel_inds = np.argwhere(skel > 0)
    overlay = label * 50 + edges * 100
    output = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bp_density = []
    measurement_number = 0
    for i in range(0, len(skel_inds), 50):
        measurement_number += 1
        x = skel_inds[i][0]
        y = skel_inds[i][1]
        this_tile = bp_labels[x - 25 : x + 25, y - 25 : y + 25]
        bp_number = len(np.unique(this_tile)) - 1
        bp_density.append(bp_number)
        text_placement = [y, x]
        output = cv2.putText(
            img=output,
            text=str(measurement_number),
            org=text_placement,
            fontFace=3,
            fontScale=1,
            color=(0, 255, 255),
        )

    bp_density = np.array(bp_density)
    bp_density[bp_density < 0] = 0
    out_list = list(zip(list(range(1, measurement_number)), bp_density))
    return out_list, output


############################################################
def generate_file_list(file_list):
    accepted_file_types = ["czi", "png", "tif", "tiff"]
    reduced_file_list = []
    file_types = []
    for file in file_list:
        if "." in file:
            file_type = file.split(".")[-1]
            if file_type in accepted_file_types:
                reduced_file_list.append(file)
                file_types.append(file_type)
    return reduced_file_list, file_types


def analyze_images(images_to_analyze, file_names, settings, out_dir):
    seg_list = []
    for s, t in zip(images_to_analyze, file_names):
        seg = segment_with_settings(s, settings)
        seg_list.append(seg)
        this_file = t.split("_slice")[0]
        this_slice = t.split("_")[-1]
        suffix = "_" + this_slice + ".png"
        if os.path.exists(os.path.join(out_dir, this_file)) == False:
            os.mkdir(os.path.join(out_dir, this_file))
        vessel_labels = make_labelled_image(s, seg)
        seg = seg * 200
        output_path = os.path.join(out_dir, this_file)
        cv2.imwrite(os.path.join(output_path, "img" + suffix), s)
        cv2.imwrite(os.path.join(output_path, "label" + suffix), seg)
        cv2.imwrite(
            os.path.join(output_path, "vessel_labels" + suffix), vessel_labels
        )
    return seg_list


def analyze_single_image(image, file_name, settings, out_dir):
    seg = segment_with_settings(image, settings)
    suffix = ".png"
    file_base = file_name.split(".")[0]
    if os.path.exists(os.path.join(out_dir, file_base)) == False:
        os.mkdir(os.path.join(out_dir, file_base))
    vessel_labels = make_labelled_image(image, seg)
    seg = seg * 200
    final_path = os.path.join(out_dir, file_base)
    cv2.imwrite(os.path.join(final_path, "img" + suffix), image)
    cv2.imwrite(os.path.join(final_path, "label" + suffix), seg)
    cv2.imwrite(
        os.path.join(final_path, "vessel_labels" + suffix), vessel_labels
    )
    return seg


def save_settings(settings, path):
    with open(path, "wb") as filehandle:
        pickle.dump(settings, filehandle)
    return


def load_settings(path):
    with open(path, "rb") as filehandle:
        all_settings = pickle.load(filehandle)
    return all_settings


def cohens_kappa_segmentation(seg1, seg2):
    seg1[seg1 > 0] = 1
    seg2[seg2 > 0] = 1
    A = 0
    B = 0
    C = 0
    D = 0
    for x in range(seg1.shape[0]):
        for y in range(seg1.shape[1]):
            i = seg1[x, y]
            j = seg2[x, y]
            if i == 1 and j == 1:
                A += 1
            if i == 1 and j == 0:
                B += 1
            if i == 0 and j == 1:
                C += 1
            if i == 0 and j == 0:
                D += 1
    P = (A + D) / (A + B + C + D)
    Pe = (A + B) * (A + C) / (A + B + C + D) ** 2 + (C + D) * (B + D) / (
        A + B + C + D
    ) ** 2
    k = (P - Pe) / (1 - Pe)
    return k


#####################################################################
# DEPRECATED


def segment_vessels(image, k=12, hole_size=500, ditzle_size=750, bin_thresh=2):
    image = cv2.medianBlur(image.astype(np.uint8), 7)
    image, background = subtract_background_rolling_ball(
        image,
        400,
        light_background=False,
        use_paraboloid=False,
        do_presmooth=True,
    )
    im_vector = image.reshape((-1,)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(
        im_vector, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    center = center.astype(np.uint8)
    label_im = label.reshape((image.shape))
    seg_im = np.zeros_like(label_im)
    for i in np.unique(label_im):
        seg_im = np.where(label_im == i, center[i], seg_im)

    _, seg_im = cv2.threshold(
        seg_im.astype(np.uint16), bin_thresh, 255, cv2.THRESH_BINARY
    )

    _, seg_im = fill_holes(seg_im.astype(np.uint8), hole_size)
    seg_im = remove_small_objects(seg_im, ditzle_size)

    return seg_im
