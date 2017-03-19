import cv2
import sys
import numpy as np


def funky_edge_detection(img):

    # go through each pixel, while drawing edges on another canvas; 
    # if it is neighboring some black pixel, it means it's edge; mark it as white; 
        # mark the middle ones as black;
    # mark the edge with a thickness of 2 pixels

    print "entered"
    width, height = img.shape[:2]   

    result_canvas = np.zeros((height, width), np.uint8) # result canvas all black

    # if on edge, mark as white; otherwise, mark black;
    for row in range (0, height - 1):
        for col in range(0, width - 1):

            if (img[row][col] and is_on_edge(img, row, col)):
                # mark on result as edge on result canvas 

                result_canvas[row][col] = 255


    print "done"
    # cv2.imshow("here, result_canvas", result_canvas)
    return result_canvas 


# check neighboring pixels to see if the current pixel is on edge
def is_on_edge(img, row, col):
    # print "called is_on_edge"
    width, height = img.shape[:2]
    assert (row >= 0 and row < height and col >=0 and col < width), (
        "is_on_edge Error: given pixel out of frame.")

    assert (img[row][col]), "given pixel is not an edge pixel; not white"

    # up, down, left, right, if in frame; 
    if ((row != 0 and not (img[row - 1][col]))      or 
        (row != height and not (img[row + 1][col])) or
        (col != 0 and not (img[row][col - 1]))      or
        (col != width and not (img[row][col + 1]))):
        return True 
    return False



# smarter is_on_edge check; 
# would ignore 



def px_comparison(im1, im2):
    # this function will update global variables
    diff = 0
    total = 0

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




def main():
    orig = cv2.imread("datasets/30_data/roi/img_12_roi_1_traced.png")

    # need to find contours
    # convert to gray
    im = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)

    edge = funky_edge_detection(im)

    # print edge
    show = np.hstack((im, edge))
    cv2.imshow("from funky_edge_detector I wrote", show)


    # # binary threshold on selected region; 
    _, img_contours, _ = cv2.findContours(im, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)   # convert to BGR
    cv2.drawContours(img, img_contours, -1, (255,255,255), 1) # drawing all contours

    show2 = np.hstack((orig, img))
    cv2.imshow("from built-in contour finding algorithm", show2)
    # A *MUST* if use cv2.imshow to debug
    while (1):

        k = cv2.waitKey(1) & 0xFF
        if (k == 27) or (k == ord("q")): 
            print "User_int: Quit key pressed."
            break
    cv2.destroyAllWindows()




main()





