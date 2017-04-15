# 27555 Computer Vision for Microstructure 3D visualization;
#       Elim Zhang, Prof. Degraef's lab
#       Written on Feb. 14, 2017
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

# function prototype
def binary_thresh (img): return    
        # apply binary threshold on normal region of img
def histo_eqlz_mask (mask): return  
        # apply histogram equalization on twin region
def remove_noise (img): return
        # Aims to remove 1-pixel noise in the background 

# def main(img_path, mask_path)
def main(img_path, mask_path):
    ## currently hard-coded paths; later paths will be fed into main() as inputs
    # img_path = "datasets/30_data/stack_img/img_20.tif"
    # mask_path = "datasets/30_data/mask_stack/img_20.tif"
    ####

    img = cv2.imread(img_path)
    normal_region = binary_thresh(img_path, mask_path)

    (height, width) = (img.shape[0:2])
    # show = cv2.resize(normal_region, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)
    # cv2.imshow("here,normal_region ", show)

    # ----------- twin region ------------
    # twin_region = histo_eqlz_mask (img_path, mask_path)
    twin_region = canny_on_twin(img_path, mask_path)
    # cv2.imshow("twin, ", twin_region)
    ## twin_show = cv2.resize(twin_region, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    # cv2.imshow("twin_region", twin_region)
    ##1


    assert twin_region.shape ==  normal_region.shape      # need same shape for bitwise op
    result = cv2.bitwise_or(normal_region, twin_region)


    
    ### FOR DEBUGGING AND IMAGE SHOWING ONLY
    height, width = result.shape[0:2]
    # result = cv2.resize(result, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    # cv2.imshow("result", result)
    ### 
    return result   

    # num = get_file_number(img_path)
    # name = "datasets/30_data/result_w_contours/img_" + str(num) + ".tif"
    # cv2.imwrite(name, result)
    # sys.exit()



    # DEBUG: A *MUST* if use cv2.imshow to debug
    while (1):
        k = cv2.waitKey(1) & 0xFF
        if (k == 27) or (k == ord("q")): 
            print "User_int: Quit key pressed."
            break
    cv2.destroyAllWindows()
    #


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
    # cv2.imshow('mask', mask)
    mask = cv2.bitwise_not(mask)

    assert (mask.shape == img.shape), ("Error: mask and img have diff dimensions")

    
    height, width = img.shape[0:2] 


    trial = cv2.bitwise_and(img, mask)      # carve out the normal region;
    
    blur_gauss = cv2.GaussianBlur(trial, (3,3), 0) # blur to remove background noise

    # morphological transformation - gradient
    kernel = np.ones((3,3), np.uint8)
    xfm = cv2.morphologyEx(blur_gauss, cv2.MORPH_GRADIENT, kernel)
    equalz_xfm = cv2.equalizeHist(xfm)

    # cv2.imshow("morph_gradient + equalizeHist", equalz_xfm)
    thresh_val = 190
    _, equalz_xfm = cv2.threshold(equalz_xfm, thresh_val, 255, cv2.THRESH_BINARY)
    # cv2.imshow('im', equalz_xfm)

    return equalz_xfm



# new trial on twin regions;
def canny_on_twin (img_path, mask_path):
    img = cv2.imread(img_path, 0)
    mask = cv2.imread(mask_path, 0) 

    # need to cut out 2 rows and 2 cols, so mask has the same shape as img; 
    mask = np.delete(mask, 0, axis = 0)    # delete first row of mask;
    mask = np.delete(mask, -1, axis = 0)   # delete last row of mask;
    mask = np.delete(mask, 0, axis = 1)    # delete first col of mask ;
    mask = np.delete(mask, -1, axis = 1)    # delete last col of mask ;

    assert (mask.shape == img.shape), ("Error: mask and img have diff dimensions")

    height, width = img.shape[0:2] 

    # mark out only the twin regions;
    twin_region = cv2.bitwise_and(img, mask)    # ROI now selected;
    edges = cv2.Canny(twin_region, 60, 130)
    
    return edges



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

    assert (mask.shape == img.shape), ("Error: mask and img have diff dimensions")

    height, width = img.shape[0:2] 


    # mark out only the twin regions;
    twin_region = cv2.bitwise_and(img, mask)    # ROI now selected;
    # roi_show_black = cv2.resize(twin_region, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    roi_show_black = twin_region
    # cv2.imshow("roi_show_black", roi_show_black)
    # cv2.imwrite("datasets/30_data/roi/twin_black_1.tif", roi_show_black)


    equalz_black = cv2.equalizeHist(roi_show_black)

    img_b = equalz_black        # a shorter name

    # binary threshold on selected region; 
    _, img_b = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            #  first return value is threshold; not needed here; 
    _, img_contours_b, _ = cv2.findContours(img_b, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
            #   return value of findContours is (im, contours, hierarchy);
            #   only need contours, and the other 2 don't matter; 
    img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)   # convert to BGR
    cv2.drawContours(img_b, img_contours_b, -1, (255,255,255), 1) # drawing all contours
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)   # convert back to gray;


    return img_b


## For DEBUG
# helper function, to extract file number in a path string; 
def get_file_number(s):
    # loop through the string;
    num = ""
    for i in range (0, len(s) - 1):
        if (s[i]).isdigit():
            num += s[i]
    return int(num)


# ## for DEBUG
# img_path = "datasets/30_data/stack_img/img_20.tif"
# mask_path = "datasets/30_data/mask_stack/img_20.tif"
# main(img_path, mask_path)

# ####

k = 120

# img_path = "datasets/4000_results/stack_img/img_" + str(k) + ".tif" 
# mask_path = "datasets/4000_results/mask_stack/img_" + str(k) + ".tif"
# main(img_path, mask_path)





