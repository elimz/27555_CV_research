# --------------------------------------------------
# Experimenting Image Segmentation on 3D dataset 
#               with Python OpenCV 3.0
# 
# Prof. Degraef
# Elim Zhang, Version 2
#   yilinz@andrew
# Feb. 17, 2016 (last edit July, 2017)
# --------------------------------------------------
# This script contains helper function, that returns a mask for each image, 
#   using Floodfill;
# --------------------------------------------------

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def main (path): 
 
    img = cv2.imread(path) 
    height, width = img.shape[:2]   

    # step 1: try to use binary, and get the twin boundary regions? 
    #  try binary on both orig and histed pictures; 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, orig_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    

    # step 2: flood fill to create a mask;a
    #   this is a spot in the white border. 
    seed_pos = (134, 134)       # simply bc this number works for whole dataset 

    mask_orig_bin = np.zeros((height + 2, width + 2), np.uint8)
            
    cv2.floodFill(orig_bin, mask_orig_bin, seed_pos, newVal= (255, 0, 0), 
                    flags = (4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY))
                # flags: color fills only the mask, not the original image;

    return mask_orig_bin



