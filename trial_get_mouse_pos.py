
# An interactive program for debugging / testing;
# currently takes user mouse-click, & prints out the pixel values of click location

import cv2
import sys


seed = None
draw_enabled = None
im = None

# user expected to click within twin boundary region
def on_click(event, x, y, flags, param):
    # print "seed position, = ", (seed_x, seed_y)
    global seed 
    global im

    if event == cv2.EVENT_LBUTTONDOWN:
        print "clicked, (x,y) = ", (x,y)
        assert (im != None), ("im == None here")

        # drawing out the position of seed that's selected
        seed = x, y                     # update the global
        print im[x,y]

        cv2.circle(im, (x, y), 10, (255, 255, 0), -1)
    return 


def main(im_path):
    global im
    global seed 

    im = cv2.imread(im_path)
    (height, width) = im.shape[0:2]
    im = cv2.resize(im, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 

    cv2.imshow("seed set by mouse click", im)
    cv2.setMouseCallback("seed set by mouse click", on_click)
    

    while (1):
        cv2.imshow("seed set by mouse click", im)
        # cv2.setMouseCallback("seed set by mouse click", on_click)

        k = cv2.waitKey(1) & 0xFF
        if (k == 27) or (k == ord("q")): 
            print "User_int: Quit key pressed."
            break
    cv2.destroyAllWindows()

    # print seed


im_path = sys.argv[1]
main(im_path)


