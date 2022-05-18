# Vessel Metrics
Created by S.D. McGarry under the direction of S. Childs at the University of Calgary
## Description
Vessel metrics is a software package written in python designed to automate the analysis of vascular architecture. Default settings are optimized for confocal microscopy imaging with vessels on the order of 10 pixels in diameter, but will support other image types with manual user adjustment.

This project is actively under development.

## Features
### Processing raw microscopy data
Data input accepted as either native .czi files or in image format

### Image preprocessing
A variety of commonly used preprocessing techniques are included, including rolling ball background suppression, smoothing, contrast stretching and histogram equalization, and 4 vessel enhancement filters

### Vessel segmentation
Vessel segmentation included is optimized for confocal imaging. The default pipeline performs a rolling ball filter, background smoothing, a contrast stretch, and a frangi filter before binarizing the image and performing morphological operations to adjust for common artifacts. Segmentation accuracy can be verified using a jaccard index or vessel specific accuracy metrics (connectivity, area, and length)

### Vessel architecture
Default skeletonization using skimage inbuilt function is built upon to handle common artefacts in vascular imaging. Small terminal segments are removed and parent segments with multiple child branches are merged to produce a more accurate representation of vessel structure.

### Fully automatic vessel diameter calculation
Using the vessel skeleton, vessel metrics identifies individual vessel segments and branch points, determines the local slope of a segment, places a cross line of appropriate length, and measures the full width half max diameter across the vessel at each crossline. User defined crossline spacing (default every 10 pixels). 

### Vessel parameter calculation
Vessel metrics calculates branchpoint density, segment length, vessel density, and network length.

### Second channel analysis
Optimized for pericyte detection in confocal microscopy. This feature allows the integration of vessel parameters with information derived from a second image channel. 

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
