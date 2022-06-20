# Vessel Metrics
Created by S.D. McGarry under the direction of S. Childs at the University of Calgary
## Description
Vessel metrics is a software package written in python designed to automate the analysis of vascular architecture. Default settings are optimized for confocal microscopy imaging with vessels on the order of 10 pixels in diameter, but will support other image types with manual user adjustment.

This project is actively under development.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/all_diameters.png "Vessel diameters")


## Features
### Processing raw microscopy data
Data input accepted as either native .czi files or in image format

### Image preprocessing
A variety of commonly used preprocessing techniques are included, including rolling ball background suppression, smoothing, contrast stretching and histogram equalization, and 4 vessel enhancement filters.

Jerman vessel enhancement filter is included with this software package.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/raw_im.png "Raw image")

### Vessel segmentation
Vessel segmentation included is optimized for confocal imaging. The default pipeline performs a rolling ball filter, background smoothing, a contrast stretch, and a frangi filter before binarizing the image and performing morphological operations to adjust for common artifacts. Segmentation accuracy can be verified using a jaccard index or vessel specific accuracy metrics (connectivity, area, and length)

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/segmentation.png "Vessel segmentation")

### Vessel architecture
Default skeletonization using skimage inbuilt function is built upon to handle common artefacts in vascular imaging. Small terminal segments are removed and parent segments with multiple child branches are merged to produce a more accurate representation of vessel structure.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/edge_labels.png "Vessel segments")

### Fully automatic vessel diameter calculation
Using the vessel skeleton, vessel metrics identifies individual vessel segments and branch points, determines the local slope of a segment, places a cross line of appropriate length, and measures the full width half max diameter across the vessel at each crossline. User defined crossline spacing (default every 10 pixels). 

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/segment_diameter.png "Single segment diameter")

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
* cv2_rolling_ball
* bresenham
* itertools
* math

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
image = cv2.medianBlur(vessels_raw.astype(np.uint8),median_size)
vessel_preproc, background = subtract_background_rolling_ball(image, ball_size, light_background=False,
                                                            use_paraboloid=False, do_presmooth=True)
vessel_preproc = vm.contrast_stretch(vessel_preproc)
```
### Vessel segmentation
``` Python
# filter options meijering, sato, frangi, jerman
# hole size (default 50) controls the size of small holes to be filled in post segmentation
# ditzle size (default 500) removes small objects post segmetnation
# sigmas (default (1,10,2)) controls sigma values input to enhancement filter
# thresh (default 60) is the threshold value to binarize the final enhanced image
vessel_seg = vm.brain_seg(vessel_raw, filter = 'frangi', thresh = 10)

```
### Vessel architecture
``` Python
skel, edges, bp = vm.skeletonize_vm(vessel_seg)
_, edge_labels = cv2.connectedComponents(edges)
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
# network length is the summation of all segment lengths
net_length = vm.network_length(edges)

# vessel density is the number of vessel pixels vs total pixels
# 16,16 denotes how many x and y chunks to break the image into (in this case 16 and 16)
_, vessel_density = vm.vessel_density(vessel_preproc, vessel_seg, 16, 16)

bp_density = vm.branchpoint_density(skel, vessel_seg)

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
```
