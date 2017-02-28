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


# ======================================
#       Testing Single Module
# ======================================
''' 
using opencv instead
'''
def main (path): 
    global orig_bin


    img_single_channel= cv2.imread(path,0)        # this chanegs image into 2 channels only;
    img = cv2.imread(path) 
    res = cv2.equalizeHist(img_single_channel)  # after equalization
                                                # input is 8-bit single channel image 

    # scaling
    width, height = res.shape[:2]   # take first 2 componenets of tuple for h, w

    res = cv2.resize(res, 
                    (height / 2, width / 2), 
                    interpolation = cv2.INTER_CUBIC) 
    img = cv2.resize(img, 
                    (height / 2, width / 2), 
                    interpolation = cv2.INTER_CUBIC) 


    # step 2: try to use binary, and get the twin boundary regions? 
    #  try binary on both orig and histed pictures; 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh_1, orig_bin) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    (thresh_1, res_bin) = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)


    # step 3: find contour for the twin boundary    
    # 3.1. original image in binary; 
    ret_orig, thresh_orig = cv2.threshold(orig_bin, 127, 255, 0)
    im, orig_contours, hierarchy = cv2.findContours(orig_bin, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    orig_bin = cv2.cvtColor(orig_bin, cv2.COLOR_GRAY2BGR)   # convert to BGR
    cv2.drawContours(orig_bin, orig_contours, -1, (255,255,255), 1) # drawing all contours

    # 3.2. image after histogram equalization; 
    ret_res, thresh_res = cv2.threshold(res_bin, 127, 255, 0)
    im, res_contours, hierarchy = cv2.findContours(res_bin, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    res_bin = cv2.cvtColor(res_bin, cv2.COLOR_GRAY2BGR)   # convert to BGR
    cv2.drawContours(res_bin, orig_contours, -1, (255,255,255), 1) # drawing all contours



    # A *MUST* if use cv2.imshow to debug
    while (1):
        # show a BGR style of orig_bin;
        show = np.hstack((orig_bin, res_bin))
        cv2.imshow("original binary vs. histogram binary ", show)

        k = cv2.waitKey(1) & 0xFF
        if (k == 27) or (k == ord("q")): 
            print "User_int: Quit key pressed."
            break
    cv2.destroyAllWindows()
    return



# path: start without "/" while entering path
# path = ("datasets/30_data/stack_img/img_5.tif")
main(path = sys.argv[1])










# ======================================
#       When called by top_module
# ======================================
# ''' 
# using opencv instead
# '''
# def histogram (path):
#     img = cv2.imread(path,0)
#     res = cv2.equalizeHist(img)
#     # show = np.hstack((img,res)

#     # scaling
#     width, height = res.shape[:2]   # take first 2 componenets of tuple for h, w

#     # res = cv2.resize(res, 
#     #                 (height / 2, width / 2), 
#     #                 interpolation = cv2.INTER_CUBIC) 

#     # cv2.imshow("result after histogram equalization", res) 

#     return res;
    




