#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:54:03 2021

autonomous crack detection library

@author: sean
"""
import numpy as np
from scipy import ndimage
import cv2
import fractions
import math
import random
from skimage import morphology,feature

def get_skel_points(img,cropped = True):
    """
    This function obtains positive cracks from the given image.

    Arguments:
    img - The input image
    
    Return:
    x - x coordinates
    y - y coordinates
    
    """
    
    (y,x) = np.nonzero(img) # Returns a tuple of array ([y],[x])
    
    # True only if coordinate shifting, False if using global coordinates
    if cropped:
        # Coordinate shifting to have center at (0,0) and flip the y-axis
        y = -1*(y - 1) # Flips the y axis
        x = x - 1        
    
    return x,y

def get_dydx(cropped_img,linspace = 51):
    """
    This function gets a 3x3 cropped image to and approximates the 
    crack points as a polynomial. The x and y values are placed on
    a grid as shown:
    
    [[(-1,1),(0,1),(1,1)],
    [(-1,0),(0,0),(1,0)],
    [(-1,-1),(0,-1),(1,-1)]]
    
    The point of interest is only at the gradient at point (0,0)
    
    Arguments:
    x - x coordinates
    y - y coordinates
    space = space for linspace
    """
    
    # Check if there is only 1 pixel in crack, then return early
    
    if np.sum(np.squeeze(cropped_img.flatten())) == 1:
        plt.figure()
        plt.imshow(cropped_img,extent = [-3/2.,3/2., -3/2.,3/2.])
        return 0, False
    # Obtain the x and y value of cropped image
    x,y = get_skel_points(cropped_img)
    #print(x,y)
    num_points = len(x)
    vert = False   
    # Checks if all values of y
    x_check = [-1,0,1]
    
    # Check for possible vertical 
    for x_val in x_check:
        occurences = np.count_nonzero(x == x_val)
        if occurences >= 2 and num_points <= 3 :
            #print(occurences)
            vert = True
            break          
    
    p_deg = (num_points - 1) if num_points > 1 else 1
    linspace = (linspace + 1) if linspace % 2 == 0 else linspace
    x_new = np.linspace(-1,1,num = linspace)
    
    
    #print(x_new)
    if vert == False:
        plt.figure()
        plt.imshow(cropped_img,extent = [-3/2.,3/2., -3/2.,3/2.])
        weights = np.polyfit(x,y,p_deg)
        fx = np.poly1d(weights)
    else:
        plt.figure()
        plt.imshow(cropped_img,extent = [-3/2.,3/2., -3/2.,3/2.])
        cropped_img = np.rot90(cropped_img) # Rotate the image anticlockwise 90 degrees
        x,y = get_skel_points(cropped_img) # Update the x,y coordinates
        weights = np.polyfit(x,y,p_deg)
        fx = np.poly1d(weights)        
    
    y_new = fx(x_new)
    x0_pos = np.where(x_new == 0)
    #print(y_new)
    dy = np.gradient(y_new,2/(linspace-1))
    #print(dy)
    #print(x0_pos)
    dy0 = str(dy[x0_pos])[1:-1] # dy when x = 0 stored in string format
    
    dy = fractions.Fraction(dy0).limit_denominator()
    
    plt.figure()
    plt.imshow(cropped_img,extent = [-3/2.,3/2., -3/2.,3/2.])
    plt.plot(x,y,"bo")
    plt.plot(x_new,fx(x_new))
    plt.show()
    return dy,vert


def get_cropped(img,x,y):
    """
    This function returns 3x3 cropped images at every 
    skeleton point
    
    Arguments:
    img - the skeleton image
    x - x global coordinate of the skeleton
    y - y global coordinate of the skeleton
    """
     
    (row,column) = np.meshgrid(np.array([x-1,x,x+1]), np.array([y-1,y,y+1]))
    row = row.astype('int')
    column= column.astype('int')
    
    cropped_img = img[column,row]
    
    return cropped_img

def get_imp_points(img):

    end_points = []
    inter_points = []
    #img = np.pad(img,((1,1),(1,1)),"constant")
    (rows,cols) = np.nonzero(img)
    
    for (r,c) in zip(rows,cols):

        if np.sum(img[r-1:r+2,c-1:c+2]) == 2:
            end_points.append((c,r))
        elif np.sum(img[r-1:r+2,c-1:c+2]) > 3:
            inter_points.append((c,r))
     
    #     for point1 in t_inter_points:
    #         temp_x = [point1[0]]
    #         temp_y = [point1[1]]
    #         for point2 in t_inter_points:
    #             if ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 < 9**2) and (point1 != point2):
    #                 temp_x.append(point2[0])
    #                 temp_y.append(point2[1])

    #         x_avg = int(math.floor(sum(temp_x)/len(temp_x)))
    #         y_avg = int(math.floor(sum(temp_y)/len(temp_y)))

    #         inter_points.append((x_avg,y_avg))  

    #     #all_branch = get_branches(img)
    #     end_points = list(set(end_points))
    #     inter_points = list(set(inter_points))
    #     #num_branch = get_num_branch(all_branch,inter_points)
    
    return end_points, inter_points

def get_branches(img):
    """
    This algorithm gets the branches of the skeleton image from end points to
    intersection point. The determined branches are used for DSE pruning method.
    
    Branches from one intersection point to another intersection point is not considered
    because it is definitely important to the reconstruction of the binary image.
    
    i.e. Only end point to intersection point is considered. 
    
    Arguments:
    img - Skeletonized image
    
    Return
    all_branches - A list of lists of coordinates (x,y) of the branches
    """
   
    E,_ = get_imp_points(img)  
    all_branches = []
    
    #print(f"The end points: {E}")
    for e in E:
        branch = []
        r_g, c_g = e[1], e[0]
        branch.append((c_g,r_g))

        while np.sum(img[r_g-1:r_g+2,c_g-1:c_g+2]) <= 3:
            (r_t,c_t) = np.nonzero(img[r_g-1:r_g+2,c_g-1:c_g+2])
            r_t = r_t-1+r_g
            c_t = c_t-1+c_g
            for point in zip(c_t,r_t):
                if point not in branch:
                    branch.append(point)

            r_g = branch[-1][1]
            c_g = branch[-1][0]
            if (c_g,r_g) in E:
                branch.append((c_g,r_g))
                break
        all_branches.append(branch)
    
    
    # Fit a Gaussian distribution for the list of branch
#     len_branches = [len(branch) for branch in all_branches]
#     print(f"Length of all branches: {len_branches}")
#     avg_len = np.ceil(np.mean(len_branches))
#     print(f"The average length: {avg_len}")
#     std_len = np.std(len_branches)
#     print(f"The s.deviation: {std_len}")
    
#     g = lambda x:np.exp(-(x-avg_len)**2/(2*std_len**2))
    
#     plt.hist(len_branches)
#     w_avg_len = np.mean(([g(i)*i for i in len_branches]))
#     print(f"The weighted avg len: {w_avg_len}")
#     all_branches = [branch for branch in all_branches if len(branch) <= avg_len] 
    
    return all_branches

def get_avg_curve_len(img):
    """
    This algorithm gets the normalized curve length
    
    Branches from one intersection point to another intersection point is not considered
    because it is definitely important to the reconstruction of the binary image.
    
    i.e. Only end point to intersection point is considered. 
    
    Arguments:
    img - Skeletonized image
    
    Return
    all_branches - A list of lists of coordinates (x,y) of the branches
    """
       
    all_branch_len = []
    s = img.copy()*1
    i = 0
    
    while True:

        all_branch = get_branches(s)
        for branch in all_branch:
            all_branch_len.append(len(branch))
            r = [i[1] for i in branch]
            c = [i[0] for i in branch]

            s[r,c] = 0
        if i != 0:
            all_branch_len.append(np.sum(s))
                
        if len(all_branch) == 0:
            break
        
        i += 1
    try:
        avg_len = np.mean(all_branch_len)
    except:
        return 0
    return avg_len



def get_num_branch(all_branches,inter_points):
    
    list_num_branch = []
    for point in inter_points:
        num_branch = 0
        for branch in all_branches:
            if point in branch:
                all_branches.remove(branch)
                num_branch += 1
        list_num_branch.append(num_branch)
        
    return list_num_branch

def reconstruct(skel_img,dist_tr):
    """
    Attempt to reconstruct the binary image from the skeleton
    
    Arguments:
    img - Skeleton image using thinning algorithm
    dist_tr - Distance transform matrix
    
    Return:
    bn_img - Binary image
    """
    row, col = np.nonzero(skel_img)
    bn_img = skel_img.copy()*1
    for (r,c) in zip(row,col):
        radius = math.ceil(dist_tr[r,c]-1)
        if radius >= 1:
            stel = morphology.disk(radius)
            bn_img[r-radius:r+radius+1,c-radius:c+radius+1] += stel
    
    return bn_img >= 1


def DSE_v1(img,threshold):
    """
    Discrete Skeletonization Evolution algorithm
    Prunes spurious branches obtained from medial axis transform.
    
    Arguments:
    img - Binary image
    
    Returns:
    pruned_img - Pruned binary image using DSE
    """

    _,dist = morphology.medial_axis(img,return_distance = True) 
    skel_img = morphology.thin(img)
    skel_img = morphology.closing(skel_img)
    ori_img = img.copy()
    all_branches = get_branches(skel_img)
    iou = lambda ori_img,img2: np.sum(ori_img*img2)/np.sum((ori_img+img2)>=1)

    iou_scores = []
    for branch in all_branches:
        skel_removed = skel_img.copy() # Initialization
        r = [i[1] for i in branch]
        c = [i[0] for i in branch]
        skel_removed[r,c] = 0

        bn_removed = reconstruct(skel_removed,dist) # Reconstruct the binary image without the branch 
        iou_scores.append(iou(ori_img,bn_removed))
        
    for score in reversed(sorted(iou_scores)):
        idx = iou_scores.index(score)
        
        if score > threshold:
            for points in all_branches[idx]:
                r,c = points[1], points[0]
                skel_img[r,c] = 0
        else:
            return skel_img, reconstruct(skel_img,dist)
        
    return morphology.closing(skel_img), reconstruct(skel_img,dist)

def DSE_v2(img,threshold):
    """
    Discrete Skeletonization Evolution algorithm
    Prunes spurious branches obtained from medial axis transform.
    
    Arguments:
    img - Binary image
    
    Returns:
    pruned_img - Pruned binary image using DSE
    """

    _,dist = morphology.medial_axis(img,return_distance = True) 
    skel_img = morphology.thin(img)
    skel_img = morphology.closing(skel_img)
    all_branches = get_branches(skel_img)
    iou = lambda ori_img,img2: np.sum(ori_img*img2)/np.sum((ori_img+img2)>=1)

    while True:
        
        iou_scores = []
        
        for branch in all_branches:
            skel_removed = skel_img.copy() # Initialization
            r = [i[1] for i in branch]
            c = [i[0] for i in branch]
            skel_removed[r,c] = 0
            bn_removed = reconstruct(skel_removed,dist)
            iou_scores.append(iou(img,bn_removed))
            
        if len(iou_scores)>0:
            max_score = max(iou_scores)
            if max_score > threshold:
                idx = iou_scores.index(max_score)

                for points in all_branches[idx]:
                    r,c = points[1], points[0]
                    skel_img[r,c] = 0
                del all_branches[idx]

            else:
                return skel_img, reconstruct(skel_img,dist)
        else:
            return skel_img, reconstruct(skel_img,dist)
        
def DSE_v3(img,beta):
    """
    Discrete Skeletonization Evolution algorithm which finds the trade 
    off between skeleton simplicity and reconstruction error.
    
    Arguments:
    img - Binary image
    
    Returns:
    pruned_img - Pruned binary image using DSE
    """    
    
    M,dist = morphology.medial_axis(img,return_distance = True)
    #M = morphology.skeletonize(img)
    S_all = [M]
    norm_dist = lambda s: np.log(np.sum(s)+1)
    norm_area = lambda s,d: (np.sum(d)-np.sum(reconstruct(s,dist)))/(np.sum(d)+1)
    
    all_branches = get_branches(M)
    avg_curve_len = get_avg_curve_len(M)
    all_branches = [branch for branch in all_branches if len(branch) < np.ceil(avg_curve_len)]

    M_len = norm_dist(get_avg_curve_len(M))
    avg_branch_len = np.mean([len(branch) for branch in all_branches]) # Average length of all branches
    alpha = beta * np.log(avg_branch_len/M_len+1)
    scores = [alpha*norm_area(M,reconstruct(M,dist))+norm_dist(M)]
    
    while len(all_branches) > 2:
        weights = []
        for branch in all_branches:
            S = M.copy()                  # S_(i) 
            r = [i[1] for i in branch]
            c = [i[0] for i in branch]

            S[r,c] = 0                    # Initialize the weights
            AR = norm_area(img,reconstruct(S,dist))
            LR = norm_dist(get_avg_curve_len(S))
            weights.append(alpha*AR + LR)

        min_idx = np.argmin(weights)
        E = all_branches[min_idx]         # The minimum branch to be removed 
        r = [i[1] for i in E]             # Get the rows of minimum weight branch
        c = [i[0] for i in E]             # Get the columns of minimum weight branch
        M[r,c] = 0                        # Remove the minimum branch from the medial axis, S_(i+1)
        AR = norm_area(img,reconstruct(M,dist))
        LR = norm_dist(get_avg_curve_len(M))

        S_all.append(M)
        del all_branches[min_idx]    
    
    if len(S_all) >= 2:
        print(f"Length of S_all: {len(S_all)}")
        for S in S_all:
            AR = norm_area(img,reconstruct(S,dist))
            LR = norm_dist(get_avg_curve_len(S))
            scores.append(alpha*AR + LR)
            S_best = S_all[np.argmin(scores)]
            S_best = morphology.binary_dilation(S_best,morphology.disk(2))
            S_best = morphology.thin(S_best)

            return S_best, reconstruct(S_best,dist)
    
    else:
        print(f"Defaulting to DSE_v2")
        return DSE_v2(img,0.9)      



