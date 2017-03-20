# 27555 Computer Vision for Microstructure 3D visualization;
# Elim Zhang, A project under Prof. Degraef
# Written Feb. 14, 2017
#
# part of helper functions, called by top_module.py
#
# current functions: 
#       - 


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

    # step 2: flood fill to create a mask;
    seed_pos = (150, 150)
    mask_orig_bin = np.zeros((height + 2, width + 2), np.uint8)
    # mask_inv = cv2.bitwise_not(mask_orig_bin)
    # cv2.imshow("mask_inv", mask_inv)
        # flags set the color used to fill only the mask, not the image;
    cv2.floodFill(orig_bin, mask_orig_bin, seed_pos, newVal= (255, 0, 0), 
                    flags = (4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY))
    
    # RETURN VALUE: currently onily returns the mask from orig binary img
    return mask_orig_bin




# path = sys.argv[1]
main("datasets/30_data/stack_img/img_12.tif")



