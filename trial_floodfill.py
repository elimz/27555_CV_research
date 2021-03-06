
# Using floodfill to generate masks for the binary images
# has the potential to let user set seed position for floodfill;


'''
Experimenting with floodfill; 
set mask to be the entire image; 

'''
import cv2
import numpy as np 
import sys

# Declaring Global Variables
seed = None
draw_enabled = None
img = None


# mouse callback function; 
# allows user to click on image, to determine position of flood fill seed; 
# user expected to click within twin boundary region
def on_click(event, x, y, flags, param):
    # print "seed position, = ", (seed_x, seed_y)
    global seed 
    global draw_enabled

    if event == cv2.EVENT_LBUTTONDOWN:
        print "clicked, (x,y) = ", (x,y)
        # drawing out the position of seed that's selected
        seed = x, y                     # update the global
        draw_enabled = 1                # draw enabled 

        cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
    return 

# function returns the mask 

# def main(path_img):
def flood_fill(path_img, seed_pos):
    global seed          # seed position modified by on_click function
    global draw_enabled       # only use flood fill if draw_enabled is 1;
    global img 

    seed = (0, 0)
    draw_enabled = 0

    img = cv2.imread(path_img)
    h, w = img.shape[:2]            # img.shape returns (height, width)
                                    # TODO: update other files

    # setting seed position

    mask = np.zeros((h + 2, w + 2), np.uint8)
    # scaling
    # width, height = img.shape[:2]   # take first 2 componenets of tuple for h, w
    # img = cv2.resize(img, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
    # mask = np.zeros((h/2 + 2, w/2 + 2), np.uint8)       # initializing mask;
    # mask = cv2.resize(mask, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)  

    while (1):

        ###### hardcoded floodfill seed; ######

        # flags set the color used to fill the mask;
        cv2.floodFill(img, mask, seed_pos, newVal= (255, 255, 0), 
                    flags = (4 | (255 << 8) ))
                       # flags = (4 | (255 << 8)| cv2.FLOODFILL_MASK_ONLY))
        # img_show = cv2.resize(img, (h / 2, w / 2), interpolation = cv2.INTER_CUBIC) 
        cv2.imshow("img", img)
        # mask_show = cv2.resize(mask, (h / 2, w / 2), interpolation = cv2.INTER_CUBIC) 
        cv2.imshow("mask", mask)


        ###### TODO: trial: user input, mouse click for floodfill seed; ######
        # # if draw_enabled:
        # #     cv2.floodFill(img, mask, (0, 0), newVal= (255, 0, 0), 
        # #                flags = (cv2.FLOODFILL_MASK_ONLY | 4 | (1 << 8)))
        # #     draw_enabled = 0
        # # cv2.floodFill(img, mask, seed, newVal = None, 
        # #             flags = ( 4 | (1 << 8)))
        #                     # flags = (cv2.FLOODFILL_MASK_ONLY | 4 | (1 << 8)))
        #         #  consider 4 nearest neighbors, with color 25
        # # cv2.FLOODFILL_MASK_ONLY

        # # showing the img and floodfill when updated;
        # # show = np.hstack((img,mask))
        # # cv2.imshow("flood fill trial - seed set by mouse click; with mask", img)
        # # cv2.imshow("mask", mask)
        # # cv2.imshow("img", img)
        # cv2.setMouseCallback("flood fill trial - seed set by mouse click; with mask", on_click)
        # cv2.imshow("flood fill trial - seed set by mouse click; with mask", img)
        # cv2.imshow("mask", mask)
        ###### end trial: user input, mouse click for floodfill seed; ######


        # program getting killed here; 
        k = cv2.waitKey(1) & 0xFF
        if (k == 27) or (k == ord("q")): 
            print "User_int: Quit key pressed."
            break
    cv2.destroyAllWindows()

# main(sys.argv[1])
# flood_fill("screenshots/contour_diff_hierarchies/contours_img5_hierachy_-1.tif", (60,60))



