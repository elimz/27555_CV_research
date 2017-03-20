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
def main():
    ## currently hard-coded paths; later paths will be fed into main() as inputs
    img_path = "datasets/30_data/stack_img/img_12.tif"
    mask_path = "datasets/30_data/mask_stack/img_12.tif"
    
    img = cv2.imread(img_path)
    height, width = img.shape[0:2]

    normal_region = binary_thresh (img_path, mask_path)
    # do opening on normal_region, to reduce noises in the background;
    kernel_3_sq = np.ones((3,3), np.uint8)

    normal_closing = cv2.morphologyEx(normal_region, cv2.MORPH_CLOSE, kernel_3_sq)

    # cv2.imshow("normal_closing, kernel (3,3) closing ", normal_closing) 
    normal_region = normal_region[0:300, :]
    cv2.imshow("normal_region", normal_region)
    denoise = remove_noise(normal_region)
    # cv2.imwrite("denoise.png", denoise)
    # denoise = denoise[0 : 300, : ]
    cv2.imshow("denoise", denoise)

    # normal_show = cv2.resize(normal_closing, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    # cv2.imshow("normal_region", normal_show)



    # ----------- twin region ------------
    twin_region = histo_eqlz_mask (img_path, mask_path)
    ## twin_show = cv2.resize(twin_region, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    ## cv2.imshow("twin_region", twin_show)

    
    result = twin_region ##1
    
    # print twin_region.shape, normal_region.shape
    ##1 result = cv2.bitwise_or(normal_region, twin_region)
    
    ### FOR DEBUGGING AND IMAGE SHOWING ONLY
    height, width = result.shape[0:2]
    result = cv2.resize(result, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    # cv2.imshow("combining twin and normal regions", result)


    # cv2.imwrite("datasets/30_data/roi/img_12_roi_comb.tif", result)

    # A *MUST* if use cv2.imshow to debug
    while (1):

        k = cv2.waitKey(1) & 0xFF
        if (k == 27) or (k == ord("q")): 
            print "User_int: Quit key pressed."
            break
    cv2.destroyAllWindows()




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

    
    height, width = img.shape[0:2] 


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

    assert (mask.shape == img.shape), ("Error: mask and img have diff dimensions")

    height, width = img.shape[0:2] 

    # mark out only the twin regions;
    twin_region = cv2.bitwise_and(img, mask)    # ROI now selected;
    ## roi_show_black = cv2.resize(twin_region, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    roi_show_black = twin_region

    # select the normal region, and mark it as white; 
    inv_mask = cv2.bitwise_not(mask)
    roi_show_white = cv2.bitwise_or(twin_region, inv_mask) # normal region selected; 
                                                # turn into white; 
    ## roi_show_white = cv2.resize(roi_show_white, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 

    print roi_show_black.shape == roi_show_white.shape
    show1 = np.hstack((roi_show_black, roi_show_white))
    # cv2.imshow("ROI comparison: black bkg, white bkg",show1)

    


    # TODO: want to change the unselected region into white, 
        # might make equalizeHist result better
    # twin_region = cv2.equalizeHist(unselected) 
    equalz_black = cv2.equalizeHist(roi_show_black)
    equalz_white = cv2.equalizeHist(roi_show_white)



    ### Plotting; for progress report only.
    show2 = np.hstack((equalz_black, equalz_white))
    ## cv2.imshow("equalz_black", equalz_black)

    # cv2.imshow("Equalization results: black bkg, white bkg",show2)


    
    img_b = equalz_black
    img_w = equalz_white

    # binary threshold on selected region; 
    _, img_b = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            #  first return value is threshold; not needed here; 
    _, img_contours_b, _ = cv2.findContours(img_b, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
            #   return value of findContours is (im, contours, hierarchy);
            #   only need contours, and the other 2 don't matter; 
    img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)   # convert to BGR
    cv2.drawContours(img_b, img_contours_b, -1, (255,255,255), 1) # drawing all contours
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)   # convert back to gray;


    # same as above block
    _, img_w = cv2.threshold(img_w, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            #  first return value is threshold; not needed here; 
    _, img_contours_w, _ = cv2.findContours(img_w, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
            #   return value of findContours is (im, contours, hierarchy);
            #   only need contours, and the other 2 don't matter; 
    img_w = cv2.cvtColor(img_w, cv2.COLOR_GRAY2BGR)   # convert to BGR
    cv2.drawContours(img_w, img_contours_w, -1, (255,255,255), 1) # drawing all contours
    img_w = cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY)   # convert back to gray;

        
    show3 = np.hstack((img_b, img_w))
    ## cv2.imshow("img_b", img_b)
    # cv2.imshow('contours: black bkg, white bkg', show3)

    ### FOR DEBUGGING AND VIEWING ONLY 
    # img = cv2.resize(img, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    # mask = cv2.resize(mask, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    # equal_1 = cv2.resize(equal_1, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 

    # print "mask.shape = ", mask.shape, "img.shape = ", img.shape
    # assert (mask.shape == img.shape), ("Error 2: mask and img have diff dimensions")
    

    return img_b

# # remove image by scaling down and up;
# def remove_noise(img):
#     width, height = img.shape[0:2]

#     kernel = np.ones((2,2), np.uint8)

#     erode_1 = cv2.erode(img, kernel, iterations = 1)
#     erode_1 = erode_1[0 : 300, : ]
#     cv2.imshow("erode_1, kernel (2,2), 1 iteration", erode_1)

#     erode_2 = cv2.erode(img, kernel, iterations = 2)
#     erode_2 = erode_2[0 : 300, : ]
#     cv2.imshow("erode_2, 2 iterations", erode_2)
    
#     # bigger = cv2.resize(img, (height , width), interpolation = cv2.INTER_CUBIC) 
#     # cv2.imshow("bigger", bigger)
#     result = erode_1
#     return result 



# remove single-pixel noises in the background
def remove_noise (img): 
    height, width = img.shape[0:2]  ##TODO: fix everything else
    # height /= 5
    # height = 300 ## hardcoded currenly

    # print "here----", width, height

    for row in range (0, height):
        for col in range (0, width):

            if (img[row][col]):
                # print "this is a white pixel"
                if is_small_pixel(img, row, col):
                # reset it to be black;
                    # print "== here"
                    img[row][col] = 0
                    # print "now =", img[row][col]
    print "done removing noises"
    return img
        # Aims to remove 1-pixel noise in the background 

def is_small_pixel(img, row, col):
    # check up down left right, UR UL, DR, DL;
    dcol = [-1, 0, 1]
    drow = [-1, 0, 1]

    total_neigh = 0

    for i in range (0, 3):
        for j in range (0, 3):
            row += drow[i]
            col += dcol[j]
            if is_in_frame(img, row, col):
                # check wether it has a neighbor; 
                # if less than 1 neighbor, return true. 
                ## TODO: here 1 neighbor is hardcoded;
                if (img[row][col]):
                    total_neigh += 1
    
    if (total_neigh < 1):
        print "found small pixel at ", row, col
        return True

    return False


def is_in_frame(img, row, col):
    height, width = img.shape[0:2]

    if ((row < 0) or
        (row >= height) or
        (col < 0) or
        (col >= width)):
        return False

    return True

main()


