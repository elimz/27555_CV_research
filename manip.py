# --------------------------------------------------
# Experimenting Image Segmentation on 3D dataset 
#               with Python OpenCV 3.0
# 
# Prof. Degraef
# Elim Zhang, Version 2
#   yilinz@andrew
# Feb. 17, 2016 (last edit July, 2017)
# --------------------------------------------------
# This script contains helper function, that runs different segmentation methods
#   on different regions of the imagel
#   - For normal region: it uses binary threshold, currently set at 190;
#   - For twin region: it uses Canny edge detection  
# --------------------------------------------------


import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# function prototype
def binary_thresh (img): return     # segmentation for the normal region 
def histo_eqlz_mask (mask): return  # segmentation for the twin region 

# Main: 
#   reads in an image and a mask and returns a segmented image
def main(img_path, mask_path):
    img = cv2.imread(img_path)
    
    # ----------- segment normal region ------------
    normal_region = binary_thresh(img_path, mask_path)
    (height, width) = (img.shape[0:2])

    # ----------- segment twin region ------------
    twin_region = canny_on_twin(img_path, mask_path)
    assert twin_region.shape ==  normal_region.shape      # need same shape for bitwise op
    result = cv2.bitwise_or(normal_region, twin_region)

    return result   


# For normal region, use Morphological gradient functino to find the contour, 
#   use histogram equalization and binary threshold to adjust contrast and
#   remove noise;
# input: image and its mask (twin and grain boundaries are in mask)
# output: normal image regions after binary threshold; with twin and grain 
#   boundaries regions untouched; 
def binary_thresh (img_path, mask_path):

    img = cv2.imread(img_path, 0)
    mask = cv2.imread(mask_path, 0) 

    # need to cut out 2 rows and 2 cols, so mask has the same shape as img; 
    mask = np.delete(mask, 0, axis = 0)    # delete first row of mask;
    mask = np.delete(mask, -1, axis = 0)   # delete last row of mask;
    mask = np.delete(mask, 0, axis = 1)    # delete first col of mask ;
    mask = np.delete(mask, -1, axis = 1)    # delete last col of mask ;

    # need to invert mask; currently dark regions (== 0) is the normal regions
    mask = cv2.bitwise_not(mask)

    assert (mask.shape == img.shape), ("Error: mask and img have diff dimensions")

    region = cv2.bitwise_and(img, mask)      # carve out the normal region;
    blur_gauss = cv2.GaussianBlur(region, (3,3), 0) # blur to remove background noise

    # morphological transformation - gradient
    kernel = np.ones((3,3), np.uint8)
    xfm = cv2.morphologyEx(blur_gauss, cv2.MORPH_GRADIENT, kernel)
    equalz_xfm = cv2.equalizeHist(xfm)

    thresh_val = 190
    _, equalz_xfm = cv2.threshold(equalz_xfm, thresh_val, 255, cv2.THRESH_BINARY)

    return equalz_xfm


# For twin region, run Canny edge detection to segment image; 
def canny_on_twin (img_path, mask_path):
    img = cv2.imread(img_path, 0)
    mask = cv2.imread(mask_path, 0) 

    # need to cut out 2 rows and 2 cols, so mask has the same shape as img; 
    mask = np.delete(mask, 0, axis = 0)     # delete first row of mask;
    mask = np.delete(mask, -1, axis = 0)    # delete last row of mask;
    mask = np.delete(mask, 0, axis = 1)     # delete first col of mask ;
    mask = np.delete(mask, -1, axis = 1)    # delete last col of mask ;

    assert (mask.shape == img.shape), ("Error: mask and img have diff dimensions")

    # mark out only the twin regions;
    twin_region = cv2.bitwise_and(img, mask)    # ROI now selected;
    edges = cv2.Canny(twin_region, 60, 130)
    
    return edges


# histo_eqlz_mask : apply histogram equalization and Binary and Otsu thresholds
#   to a mask, to get better contrasts; 
# input: a mask that only contains the twin regions; 
# output: a mask with better contrasts;
def histo_eqlz_mask (img_path, mask_path):

    # run histogram equalization on masked region only, returns masked region 
    img = cv2.imread(img_path, 0)
    mask = cv2.imread(mask_path, 0) 

    # need to cut out 2 rows and 2 cols, so mask has the same shape as img; 
    mask = np.delete(mask, 0, axis = 0)    # delete first row of mask;
    mask = np.delete(mask, -1, axis = 0)   # delete last row of mask;
    mask = np.delete(mask, 0, axis = 1)    # delete first col of mask ;
    mask = np.delete(mask, -1, axis = 1)    # delete last col of mask ;

    assert (mask.shape == img.shape), ("Error: mask and img have diff dimensions")

    # mark out only the twin regions;
    twin_region = cv2.bitwise_and(img, mask)    # ROI now selected;
    roi = cv2.equalizeHist(twin_region)

    # binary threshold on selected region; 
    _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            #  first return value is threshold; not needed here; 
    _, roi_contours, _ = cv2.findContours(roi, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
            #   return value of findContours is (im, contours, hierarchy);
            #   only need contours, and the other 2 don't matter; 
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)   # convert to BGR
    cv2.drawContours(roi, roi_contours, -1, (255,255,255), 1) # drawing all contours
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)   # convert back to gray;

    return roi



