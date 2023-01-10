# Vessel Metrics
Created by S.D. McGarry under the direction of S. Childs at the University of Calgary
## Description
Vessel metrics is a software package written in python designed to automate the analysis of vascular architecture. Default settings are optimized for confocal microscopy imaging with vessels on the order of 10 pixels in diameter, but will support other image types with manual user adjustment.

This project is actively under development. A user guide for the user interface will be available shortly.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/methods_figure_v1.png "Vessel diameters")


## Features
### Processing raw microscopy data
Data input accepted as either native .czi files or in image format

### Image preprocessing
A variety of commonly used preprocessing techniques are included, including tophat background suppression, smoothing, contrast stretching and histogram equalization, and 4 vessel enhancement filters.

A python implementation of the 2d Jerman vessel enhancement filter is included with this software package.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/raw_im.png "Raw image")

### Vessel segmentation
Vessel segmentation included is optimized for confocal imaging. The default pipeline performs a top hat filter, background smoothing, a contrast stretch, and a frangi filter before binarizing the image and performing morphological operations to adjust for common artifacts. Segmentation accuracy can be verified using a jaccard index or vessel specific accuracy metrics (connectivity, area, and length)

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/segmentation.png "Vessel segmentation")

### Vessel architecture
Default skeletonization using skimage inbuilt function is built upon to handle common artefacts in vascular imaging. Small terminal segments are removed and parent segments with multiple child branches are merged to produce a more accurate representation of vessel structure.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/edge_labels.png "Vessel segments")
![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/fix_skeleton.png "Erroneous segments removed from skeleton")

### Fully automatic vessel diameter calculation
Using the vessel skeleton, vessel metrics identifies individual vessel segments and branch points, determines the local slope of a segment, places a cross line of appropriate length, and measures the full width half max diameter across the vessel at each crossline. User defined crossline spacing (default every 10 pixels). 

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/segment_diameter.png "Single segment diameter")
![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/diam_removal.png "Erroneous idameter measurements removed")
### Vessel parameter calculation
Vessel metrics calculates branchpoint density, segment length, vessel density, and network length.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/vessel_density.png "Vessel density map")

### Second channel analysis
Optimized for pericyte detection in confocal microscopy. This feature allows the integration of vessel parameters with information derived from a second image channel. 

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/pericyte_ovl.png "Pericyte overlay")

### Visualization tools
Functions are included to aid in the visualization of the parameter maps, overlaid over the raw image for debugging or figure generation.

## Requirements
Vessel metrics is programmed in python 3.8.12 and calls the following packages:
* cv2
* numpy
* scipy
* skimage
* matplotlib
* czifile
* bresenham
* itertools
* math
* aicsimageio
* easygui
* PIL
* Pickle

## Usage
### Processing raw microscopy data
``` Python
import vessel_metrics as vm
import os
import numpy as np

data_path = 'path/to/data/'
file = 'file_name.czi'

volume = vm.preprocess_czi(data_path,file, channel = 0)
slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
vessel_raw = reslice[0]

```
### Image preprocessing
``` Python
vessel_preproc = vm.preprocess_seg(vessel_raw)

# Alternatively, this can be done manually
image = normalize_contrast(image)

image = subtract_background(image, radius = radius, light_bg = bright_background)
image = cv2.medianBlur(image.astype(np.uint8),median_size)
image = contrast_stretch(image, upper_lim = upper_lim, lower_lim = lower_lim)
```
### Vessel segmentation
``` Python
# filter options meijering, sato, frangi, jerman
# hole size (default 50) controls the size of small holes to be filled in post processing
# ditzle size (default 500) removes small objects post segmetnation
# sigma1 (default (1,10,2)) controls sigma values input to enhancement filter
# sigma2 is only used if multi_scale = True, this setting applies 2 vessel enhancement filters and sums the images.
# thresh (default 60) is the threshold value to binarize the final enhanced image
# Your sigma numbers should cover the range of vessel diameters you expect in your image. It's ideal to set the max value smaller than the biggest vessel in your image, as the filter tends to overestimate the vessel boundaries otherwise.
vessel_seg = vm.segment_image(image, filter = 'meijering', sigma1 = range(1,8,1), sigma2 = range(10,20,5), hole_size = 50, ditzle_size = 500, thresh = 60, preprocess = True, multi_scale = True)
label = cv2.imread('/path/to/label.png',0)
length, area, conn, Q = vm.cal(label.astype(np.uint8), seg.astype(np.uint8))
jaccard = vm.jaccard(label,seg)

```
### Vessel architecture
``` Python
# The skeletonize_vm function uses the skimage skeletonize function and then post processes the skeleton to fix common errors in skeletonizing vascular trees.
skel, edges, bp = vm.skeletonize_vm(vessel_seg)
_, edge_labels = cv2.connectedComponents(edges)

_, branchpoints = vm.find_branchpoints(skel)
coords, endpoints = vm.find_endpoints(edges)
```
### Fully automatic vessel diameter calculation
``` Python
# viz is an image of equal size to vessel_preproc containing binary vessel crosslines for visualization purposes
viz, diameters = vm.whole_anatomy_diameter(vessel_preproc, vessel_seg, edge_labels)

# to visualize a single segment use visualize vessel diameter
# diam_list is the diameter measured at each crossline
# mean_diam is the mean of diam_list
# segment_viz is a binary image showing crosslines for that segment
diam_list, mean_diam, segment_viz = vm.visualize_vessel_diameter(edge_labels, segment_number, vessel_seg,vessel_preproc)
    

```
### Vessel parameter calculation
``` Python
# All of the below is accomplished using the parameter_analysis function in conjuction with the user interface.

# network length is the summation of all segment lengths
net_length = vm.network_length(edges)

# vessel density is the number of vessel pixels vs total pixels
# 16,16 denotes how many x and y chunks to break the image into (in this case 16 and 16)
_, vessel_density = vm.vessel_density(vessel_preproc, vessel_seg, 16, 16)

bp_density = vm.branchpoint_density(skel, seg)

# length is a list containing the segment length for every segment in edge_labels
_, length = vm.vessel_length(edge_labels)

end_points = vm.find_endpoints(edges)
tort, _ = vm.tortuosity(edge_labels, end_points)

```
### Second channel analysis
``` Python
volume = vm.preprocess_czi(data_path,this_file, channel = 1)
slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
pericyte_raw = reslice[0]

peri_seg = np.zeros_like(pericyte_raw)
high_vals = np.zeros_like(pericyte_raw)
high_vals[pericyte_raw>75] = 1
peri_seg[(high_vals>0) & (vessel_seg>0)]=1

kernel = np.ones((3,3),np.uint8)
peri_seg = cv2.morphologyEx(peri_seg, cv2.MORPH_OPEN, kernel)

num_labels, labels = cv2.connectedComponents(peri_seg.astype(np.uint8))

unique_labels = np.array(np.nonzero(np.unique(labels))).flatten()

reduced_label = np.zeros_like(peri_seg)
for u in unique_labels:
    numel = len(np.argwhere(labels == u))
    if numel>15 and numel<500:
        reduced_label[labels == u] = 1

```
### Visualization tools
``` Python
# alpha controls opacity
# more complex overlays can be made by for example adding the segmentation and skeleton together
vm.overlay_segmentation(vessel_preproc, vessel_seg, alpha = 0.5)
vm.show_im(image)
```

### User Interface
The user interface is called using the vm_UI.py script.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/UI_fig.png "User Interface Explanation")

The recommended workflow is to operate on a single image from your dataset, optimize the parameters to your satisfaction, and then save those settings to batch process the remainder of your dataset. Settings are saved as settings.data in the selected output directory.

#### Segmentation settings
* Filter
Determines which filter will be used. Default meijering. Accepted responses: 'meijering', 'sato', 'frangi', 'jerman'. Jerman filter recommended for low contrast data. 

* Threshold

Threshold used for final segmentation. Values are 0-255, vessel metrics applies a contrast stretch to data prior to thresholding to ensure consistency across data acquired with varying microscope settings. Default value 40.
* Sigma 1 and Sigma 2

Conceptually the sigma values describe the thickness of vessels the filters enhance. This value is always expressed in pixels. Accepted input is a comma separated series Start value, End Value, Step Size. i.e. 1,8,1. The stop value should be around 80% of the maximal vessel size expected in your dataset, choosing a larger value tends to cause overestimation of vessel boundaries. 

The second sigma parameter is only used for multi scale processing, if your image contains major vessels and microvasculature both the multi scale processing produces better results. Rather than inputting a sigma value of 1,20,5 you will see better results with 1,8,1, and 10,20,5. 
* Hole size and Ditzle size

Accepted inputs are integers. Small holes (area less than the specified value) are closed in large binary objects post segmentation. Ditzle size removes binary objects less than the specified size. The 'ditzles' are usually partially formed vessels or vessels with poor signal along the periphery of the image.
* Preprocess

Whether to preprocess the data. If you've applied your own preprocessing prior to loading vessel metrics select no. Preprocessing picks up at vessel enhancement. 
* Multi scale

Whether to do a multi scale enhancement. Useful if there is a large discrepancy in vessel size within your image. 

#### CZI Processing
Vessel metrics operates on czi files and image files (tif, png, etc). If you are using a microscopy file that isn't a czi you can create a z projection of your desired thickness using Fiji or your microscopes proprietary software and load that into vessel metrics. 

You can select which channel is used for the analysis. These are integers beginning with 0 (your first channel is channel 0). You may also select how many slices are used to create a z projection. If you select a value less than the total slices in your czi multiple slices will be created and analyzed. 
#### UI Output
A directory is created for each sample analyzed. Within each directory. An unprocessed z proejction is saved as img.png. The vessel segmentation is saved as label.png. 

Vessel_labels.png overlays the vessel skeleton and segment numbers on the vessel label. If for example you wanted to know the diameter of a particularly large vessel you would open the vessel labels image and find the associated segment number and then find that entry in the vessel_diameters.txt file. 

The vessel density function breaks the image into 256 equally sized squares and outputs the number of labelled pixels over the total number of pixels in that tile. The procedure for finding vessel density within a region is identical, find the section in question on the vessel_density.png and the related entry in vessel_density.txt.
