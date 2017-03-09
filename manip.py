# 27555 Computer Vision for Microstructure 3D visualization;
# Elim Zhang, Prof. Degraef's lab
# Written Feb. 14, 2017
#
# part of helper functions, called by topMod
#
# Histogram Equalization:
# increasing image contrasts, before image segmentation;
#
# Code followed examples from OpenCV documentations on histogram Equalization;
# 
# current functions: 
#       - currently drawing contours on original binary image, and image after 
#         histogram equalization;
# Elim Zhang,

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# function declarations
def make_mask (path): return
def binary_thresh (img): return    
        # apply binary threshold on normal region of img
def histo_eqlz_mask (mask): return  
        # apply histogram equalization on twin region





def main():
    img_path = "datasets/30_data/stack_img/img_12.tif"
    mask_path = "datasets/30_data/mask_stack/img_12.tif"
    normal_region = binary_thresh (img_path, mask_path)
    twin_region = histo_eqlz_mask (img_path, mask_path)

    result = cv2.bitwise_or(normal_region, twin_region)
    
    ### FOR DEBUGGING AND IMAGE SHOWING ONLY
    # height, width = result.shape[0:2]
    # result = cv2.resize(result, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    # cv2.imshow("combining twin and normal regions", result)

    
    cv2.imwrite("datasets/30_data/roi/img_12_roi_comb.tif", result)

    # A *MUST* if use cv2.imshow to debug
    while (1):

        k = cv2.waitKey(1) & 0xFF
        if (k == 27) or (k == ord("q")): 
            print "User_int: Quit key pressed."
            break
    cv2.destroyAllWindows()


def make_mask (path): 
 
    img = cv2.imread(path) 
    width, height = img.shape[:2]   

    # step 1: try to use binary, and get the twin boundary regions? 
    #  try binary on both orig and histed pictures; 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, orig_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    # step 2: flood fill to create a mask;
    seed_pos = (150, 150)
    mask_orig_bin = np.zeros((height + 2, width + 2), np.uint8)
        # flags set the color used to fill only the mask, not the image;
    cv2.floodFill(orig_bin, mask_orig_bin, seed_pos, newVal= (255, 0, 0), 
                    flags = (4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY))
    
    # RETURN VALUE: currently onily returns the mask from orig binary img
    return mask_orig_bin



# binary_thresh: apply binary_threshold to an image; 
# input: image, and its mask (twin and grain boundaries are in mask)
# output: normal image regions after binary threshold; 
#       twin and grain boundaries regions untouched; 
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
    width, height = img.shape[0:2] 


    # binary threshold on selected region; 
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            #  first return value is threshold; not needed here; 
    _, img_contours, _ = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
            #   return value of findContours is (im, contours, hierarchy);
            #   only need contours, and the other 2 don't matter; 
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)   # convert to BGR
    cv2.drawContours(img, img_contours, -1, (255,255,255), 1) # drawing all contours
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # convert back to gray;

    # mask out normal regions of image;  
    assert (mask.shape == img.shape), ("Error: mask and img have diff dimensions")
    normal_region = cv2.bitwise_and(img, mask)

    return normal_region




# histo_eqlz_mask : apply histogram equalization to a mask, to get better contrasts; 
# input: a mask that only contains the grain boundaries, and twin boundaries 
# output: a mask, with better contrasts;
def histo_eqlz_mask (img_path, mask_path):

    # run histogram equalization on masked region only, returns masked region 
    img = cv2.imread(img_path, 0)
    mask = cv2.imread(mask_path, 0) 

    # need to cut out 2 rows and 2 cols, so mask has the same shape as img; 
    mask = np.delete(mask, 0, axis = 0)    # delete first row of mask;
    mask = np.delete(mask, -1, axis = 0)   # delete last row of mask;
    mask = np.delete(mask, 0, axis = 1)    # delete first col of mask ;
    mask = np.delete(mask, -1, axis = 1)    # delete last col of mask ;

    # no need to invert mask; 
    #       currently white regions (== 0) is the twin regions

    assert (mask.shape == img.shape), ("Error: mask and img have diff dimensions")

    width, height = img.shape[0:2] 

    # mark out only the twin regions;
    twin_region = cv2.bitwise_and(img, mask)    # ROI now selected;

    # TODO: want to change the unselected region into white, 
        # might make equalizeHist result better
    # twin_region = cv2.equalizeHist(unselected) 
    equal_1 = cv2.equalizeHist(twin_region) 
    
    img = equal_1

    # binary threshold on selected region; 
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            #  first return value is threshold; not needed here; 
    _, img_contours, _ = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
            #   return value of findContours is (im, contours, hierarchy);
            #   only need contours, and the other 2 don't matter; 
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)   # convert to BGR
    cv2.drawContours(img, img_contours, -1, (255,255,255), 1) # drawing all contours
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # convert back to gray;

   
    ### FOR DEBUGGING AND VIEWING ONLY 
    # img = cv2.resize(img, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    # mask = cv2.resize(mask, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    # equal_1 = cv2.resize(equal_1, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 


    assert (mask.shape == img.shape), ("Error: mask and img have diff dimensions")

    return img




main()





