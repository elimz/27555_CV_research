'''
27555 Research 

Elim Zhang
Mar. 8, 2017

Function: 
    Selecting three Region of Interests, crop out and save as three files; 
    manually segment these images, and use as groundtruth to compare to 
    segmented images outputed from code; 

Files input: 
    img_12.tif from stack_img folder is used to serve as groundtruth; 
    

'''

import cv2
import sys  

def main(path):

    # read in image
    im = cv2.imread(path)
    height, width = im.shape[0:2]
    # print "height = ", height, "width = ", width

    # save to path; 
    save_to = "datasets/30_data/roi/"

    # sets ROI region - hardcoded regions, for checker functions; 
    # REGION 1:
    y_lo_1, y_hi_1 = 250, 400
    x_lo_1, x_hi_1 = 350, 500
    roi_1 = im[y_lo_1 : y_hi_1, x_lo_1:x_hi_1]
    name_1 = save_to + "stack_img_12_roi_1.tif"
    cv2.imwrite(name_1, roi_1)

    # REGION 2:
    y_lo_2, y_hi_2 = 600, 750
    x_lo_2, x_hi_2 = 150, 300
    roi_2 = im[y_lo_2 : y_hi_2, x_lo_2:x_hi_2]
    name_2 = save_to + "stack_img_12_roi_2.tif"
    cv2.imwrite(name_2, roi_2)

    # REGION 2:
    y_lo_3, y_hi_3 = 600, 750
    x_lo_3, x_hi_3 = 800, 950
    roi_3 = im[y_lo_3 : y_hi_3, x_lo_3:x_hi_3]
    name_3 = save_to + "stack_img_12_roi_3.tif"
    cv2.imwrite(name_3, roi_3)

    # draw out the bounding rectangle
    cv2.rectangle(im, (x_lo_1, y_lo_1), (x_hi_1, y_hi_1), (255, 0, 0), 3)
    cv2.rectangle(im, (x_lo_2, y_lo_2), (x_hi_2, y_hi_2), (0, 255, 0), 3)
    cv2.rectangle(im, (x_lo_3, y_lo_3), (x_hi_3, y_hi_3), (255, 200, 0), 3)
    cv2.imshow("im", im)
    name_rois = save_to + "stack_img_12_rois.tif"
    cv2.imwrite(name_rois, im)


    while (1):
        k = cv2.waitKey(1) & 0xFF
        if (k == 27) or (k == ord("q")): 
            print "User_int: Quit key pressed."
            break

path = "datasets/30_data/stack_img/img_12.tif"
main(path)