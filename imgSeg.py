
# --------------------------------------------------
# Experimenting Image Segmentation on 3D dataset 
#               with Python OpenCV 3.0
# 
# Elim Zhang, Version 1
# Dec. 25, 2016
# --------------------------------------------------
# 
# Goals: 
#   - Align frames; // WarpAffine
#   - distinguish between gamma (light grey) and gamma prime phases (dark grey);
#   - handle high-contrast twin boundary, w/ diff threshold;
# 
# Next step: 
#   - Performance: change to C++ for speed;
#   - Experiment with diff. libraries; Matlab
# 
# --------------------------------------------------

import cv2
import numpy as np


# global variables for tweaking
display_scale = 2


# main function displays original and manipulated img, and wait for quit signals
def manip(path):

    raw_1 = cv2.imread(path)
    # raw_1 = cv2.imread('datasets/30_data/stack_img/img_0.tif')
    # raw_2 = cv2.imread('datasets/30_data/stack_img/img_1.tif')

    height, width = raw_1.shape[:2]   # take first 2 componenets of tuple for h, w

    # scaling
    raw_1 = cv2.resize(raw_1, 
                    (height / display_scale, width / display_scale), 
                    interpolation = cv2.INTER_CUBIC)  

    # raw_2 = cv2.resize(raw_2, 
                    # (height / display_scale, width / display_scale), 
                    # interpolation = cv2.INTER_CUBIC)  

    # manipulate images; 
    manip_1 = get_outer_edge(raw_1)
    # manip_2 = get_outer_edge(raw_2)


    # show both images; 
    # both_1 = np.hstack((raw_1, raw_2))  
    # cv2.imshow('raw_1, raw_2', both_1)

'''
### we don't want this module to wait for user interruption 

    # wait for keyboard interruption: Q key, or esc
    key_int = cv2.waitKey(0)
    if ((key_int == ord('q')) or (key_int == 27)):              
        print "User_int: Quit key pressed."
        cv2.destroyAllWindows()
''' 

 

# Align: getAffineTransform; shifts frame, so corners overlap with the first frame; 
#   - ensures: check ratio, so the picture is not distorted during this translation;   
#   - get coord of 4 corner points; 
#   - compare to first pic, and warp it so they overlay
# 

# get_outer_edge: coarse edge detection, to pick up the outer binding rectangle;
# for image alignment purposes;
def get_outer_edge(img):

    cnt_limit = 5 # num of lines drawn on screen
    cnt = 0
    top_line_y = 0 # y_coord of top line;

    # hard-coded bounding rectangle; 
    y_lo = 0
    y_hi = img.shape[0]

    x_lo = 0
    x_hi = 50

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # convert to greyscale 

    img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)    # Gaussian Blur, 
    # cv2.imshow("w Gaussian (5,5) binary thresh 128", img_gray)



    (thresh, img_gray) = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    # try drawing the contour there as well? 
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # convert orig img back to BGR, 
    disp_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)   # convert to greyscale 
    print("-- Now drawing all the contours")
    cv2.drawContours(disp_gray, contours, -1, (0,255,0), 1) # drawing all contours
    # cv2.imshow("Contour, drawn after binary thesholding", disp_gray)
    # draw out the lines found


    # (thresh, img_gray) = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #         # This shows a really good black/white contrast; 
    #         #  why these numbers tho?
    # # cv2.imshow('after thresholding', img_gray)
    return disp_gray



'''    
    # ---- after all manipulation, now sets region ----
    # sets ROI - region of image - top left; currently hardcoded; later could be UI

    # top_half = img_gray[y_lo:y_hi, :]   # use variables to make cleaner; 
    left_half = img_gray[y_lo:y_hi, x_lo:x_hi]

    # run houghline on left_half / top_half only
    # houghline takes in a binary image; ^ above might have converted it for pritning reasons
    lines = cv2.HoughLines(left_half, 
                            rho = 1,            # dist from the origin
                            theta = np.pi/180,  # angle from the origin
                            # theta = np.pi/180, 
                            threshold = 100)

    # # try drawing the contour there as well? 
    # ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img_gray, contours, -1, (0,0,255), 2) # drawing all contours
    # cv2.imshow("left", img_gray)
    # # draw out the lines found
    if (lines == None):
        print "HoughLines didn't find any lines. "
     
    else: 
        # before drawing on left_half, convert it to colored; 
        left_half = cv2.cvtColor(left_half, cv2.COLOR_GRAY2BGR)

        for line in lines:
            rho, theta = line[0][0], line[0][1]
            # print 'line rho, theta = ', (rho, theta)

            currentLine = findLineCoord(rho, theta)
            pt1 = currentLine[0]
            pt2 = currentLine[1]

            if (cnt < cnt_limit):
                cnt += 1

                # draw on top_half first, and paste it to original image;
                cv2.line(left_half, pt1, pt2, (0, 255, 0), 1)
                
    # convert to have 3 channels; 
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # paste the top_half over original image
    # img_gray[y_lo:y_hi, :] = top_half
    img_gray[y_lo:y_hi, x_lo:x_hi] = left_half

    # draw out the bounding rectangle
    cv2.rectangle(img_gray, (x_lo, y_lo), (x_hi, y_hi), (255, 0, 0), 5)
'''

    


# get line coordinates in order to draw; 
def findLineCoord(rho, theta):

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * -b)
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * -b)
    y2 = int(y0 - 1000 * a)
    return [(x1, y1), (x2, y2)]








