
'''
27555 Research

Elim Zhang
Mar. 8, 2017

check_function.py: a checker function that verifies how accurately the image 
    segmentation is, by comparing a computer-segmented image, to a hand-traced
    image that's used as the groundtruth.

Region of interests currenlty used:
    # REGION 1:
        y_lo_1, y_hi_1 = 250, 400
        x_lo_1, x_hi_1 = 350, 500
    # REGION 2:
        y_lo_2, y_hi_2 = 600, 750
        x_lo_2, x_hi_2 = 150, 300
    # REGION 2:
        y_lo_3, y_hi_3 = 600, 750
        x_lo_3, x_hi_3 = 800, 950

'''

import cv2
import numpy as np


# global variables
diff = 0
total = 0

#               ####### Helper Functions Signatures #######
def px_comparison(im1_path, im2_path):
    return # defined below 


#                   ######## Main ###########
# main: compares regions of interests (ROI) in segmented image, and the 
#       hand-traced groundtruth;
# output: a score for similarity between the calculated image and hand-traced groundtruth

def main():

    # ------------ STEP 1 ------------: 
    # load all hand-traced ROI, from this path; 
    traced_roi_1 = cv2.imread("datasets/30_data/roi/img_12_roi_1_traced.png")
    traced_roi_2 = cv2.imread("datasets/30_data/roi/img_12_roi_2_traced.png")
    traced_roi_3 = cv2.imread("datasets/30_data/roi/img_12_roi_3_traced.png")
    traced_rois = cv2.imread("datasets/30_data/roi/img_12_rois.tif")
    (height, width) = traced_rois.shape[0:2]
    traced_rois = cv2.resize(traced_rois, (height / 2, width / 2), 
                    interpolation = cv2.INTER_CUBIC)    # scale, easier to see; 
    cv2.imshow("hand-traced regions", traced_rois)      # showing the groundtruth


    # ------------ STEP 2 ------------: 
    # load img_12.tif from manip folder 
    #       this picture underwent all image manipulations already ; 
    #       then find the ROI's, and compare to hand-traced ROi;
    # REGION 1:
    y_lo_1, y_hi_1 = 250, 400
    x_lo_1, x_hi_1 = 350, 500
    # REGION 2:
    y_lo_2, y_hi_2 = 600, 750
    x_lo_2, x_hi_2 = 150, 300
    # REGION 2:
    y_lo_3, y_hi_3 = 600, 750
    x_lo_3, x_hi_3 = 800, 950

    img = cv2.imread("datasets/30_data/manip_stack/img_12.tif")
    (height, width) = img.shape[0:2]
    traced_rois = cv2.resize(traced_rois, (height * 2, width * 2), 
                    interpolation = cv2.INTER_CUBIC) 
        # scale up twice, to match size and regions set in the groundtruth image;
    
    img_roi_1 = img[y_lo_1 : y_hi_1, x_lo_1 : x_hi_1]
    img_roi_2 = img[y_lo_2 : y_hi_2, x_lo_2 : x_hi_2]
    img_roi_3 = img[y_lo_3 : y_hi_3, x_lo_3 : x_hi_3]

    # mark the ROIs on original image; 
    img_rois = img       # a duplicate;
    cv2.rectangle(img_rois, (x_lo_1, y_lo_1), (x_hi_1, y_hi_1), (255, 0, 0), 3)
    cv2.rectangle(img_rois, (x_lo_2, y_lo_2), (x_hi_2, y_hi_2), (0, 255, 0), 3)
    cv2.rectangle(img_rois, (x_lo_3, y_lo_3), (x_hi_3, y_hi_3), (255, 200, 0), 3)
    
    # scale down to show easily; 
    img_rois = cv2.resize(img_rois, (height/2, width/2), interpolation = cv2.INTER_CUBIC) 

    cv2.imshow("manip image, marked with ROIs", img_rois)
    cv2.imwrite("datasets/30_data/roi/img_12_manip_rois.tif", img_rois)
    print "Marked image now saved."


    # ------------ STEP 3 ------------: 
    # now call px_comparison to compare all three ROIs, and get a score; 
    (diff_1, total_1)= px_comparison(traced_roi_1, img_roi_1)
    (diff_2, total_2)= px_comparison(traced_roi_2, img_roi_2)
    (diff_3, total_3)= px_comparison(traced_roi_3, img_roi_3)

    total_diff = (diff_1 + diff_2 + diff_3)
    total_total = (total_1 + total_2 + total_3)

    print "----- here", (total_diff, total_total)

    result = float (total_diff) / float(total_total)
    print "Difference Percentage = ", result


    # while loop for imshow
    while (1):
        k = cv2.waitKey(1) & 0xFF
        if (k == 27) or (k == ord("q")): 
            print "User_int: Quit key pressed."
            break
    return result




#               ####### Helper Functions #######

# px_comparison: compares two images pixel by pixel; 
#   input: paths of 2 images, with same sizes and same num of channels
#   output: a float to represent % difference between the 2 images;
def px_comparison(im1, im2):
    # this function will update global variables
    global diff, total

    # # read in 2 images;
    # im1 = cv2.imread(im1_path)
    # im2 = cv2.imread(im2_path)

    # make sure inputs meet the requirements;
    assert (im1.shape == im2.shape), "Error: im1, im2 shape mismatch;"

    # loop through the images and compare pixel;
    (height, width) = im1.shape[0:2]


    for row in range(0, height):
        for col in range (0, width):
            # check the same position of pixel in both images;
            px_im1 = im1[row][col]
            px_im2 = im2[row][col]

            total += 1

            if not (np.array_equal(px_im1, px_im2)):
                # found a different pixel;
                diff += 1

    #TODO: maybe label the differences in another color?
    return (diff, total)

#              ####### Helper Functions end #######


main()







