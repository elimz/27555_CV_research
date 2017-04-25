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
    

    # step 2: flood fill to create a mask;a
    # this is a spot in the white border. 
    seed_pos = (134, 134)       # simply bc this number works for whole dataset 



    mask_orig_bin = np.zeros((height + 2, width + 2), np.uint8)
            
    cv2.floodFill(orig_bin, mask_orig_bin, seed_pos, newVal= (255, 0, 0), 
                    flags = (4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY))
                # flags: color fills only the mask, not the original image;
    



    # cv2.imshow('show mask', mask_orig_bin)

    # # DEBUG: A *MUST* if use cv2.imshow to debug
    # while (1):
    #     k = cv2.waitKey(1) & 0xFF
    #     if (k == 27) or (k == ord("q")): 
    #         print "User_int: Quit key pressed."
    #         break
    # cv2.destroyAllWindows()


    return mask_orig_bin




# path = sys.argv[1]
# main(path)
# main("datasets/4000_results/stack_img/img_181.tif")



