
# --------------------------------------------------
# Experimenting Image Segmentation on 3D dataset 
#               with Python OpenCV 3.0
# 
# Elim Zhang, Version 1
# Dec. 25, 2016
# --------------------------------------------------
# Top Module functions:
#   - read in a multi-page tiff file, after imageJ aligned the stack;
#   - histogram_equalization, to adjust picture contrast;  
#   - calls imgSeg.py
#       - draw all contours;
#       - pair-wise comparison, to distinguish between gamma and gamma' phases;
#   - saves all manipulated images into a folder - manip_stack;
#       -> feed the stack into ImageJ for a flip-book view;
# 
# --------------------------------------------------


import cv2 
from PIL import Image
import sys

# import subsidiary files: 
import imgSeg
import histo_eqlz as histo

def main():

    # reads in a stack of aligned images, as outputted by ImageJ
    stack_path = 'datasets/30_data/feb6_orig_flip.tif'
    stack_size = 30           # number of images in that stack

    # separate raw image stack; and split into separate impages; 
    num_files = split_stack(stack_path, stack_size)

    # num_test = 10            # uncomment for testing mode; 
    num_test = stack_size
    i = 0

    # call functions in other module for pair-wise comparison
    while (i < num_test):

        # call manipulation function them;
        curr_path = "datasets/30_data/stack_img/img_" + str(i) + ".tif" 
        curr_res = histo.histogram(curr_path)
        res_name = "datasets/30_data/manip_stack/img_"+ str(i) + ".tif"
        cv2.imwrite(res_name, curr_res)

        # curr_path = "datasets/30_data/manip_stack/img_"+ str(i) + ".tif"
        # curr_res = imgSeg.manip(curr_path)
        # res_name = "datasets/30_data/manip_stack/img_"+ str(i) + ".tif"
        # cv2.imwrite(res_name, curr_res)

        
        
        i += 1

    sys.exit()
    # after done with all manip, 
    # wait for keyboard interruption: Q key, or esc
    key_int = cv2.waitKey(0)
    if ((key_int == ord('q')) or (key_int == 27)):              
        print "User_int: Quit key pressed."
        cv2.destroyAllWindows()




# split multi-page tiff image (output from ImageJ after stack alignment) into 
# individual images; 
# ::Return value: num_file, number of files read; 
# save all images with name "img_<number>.tif", under folder stack_img in 
# current directory.  (modified code from Stackoverflow)

def split_stack(path, stack_size):
    num_file = 0; 

    stack = Image.open(path)
    for i in range(stack_size):
        try:
            stack.seek(i)
            stack.save('datasets/30_data/stack_img/img_%s.tif'%(i,))
            num_file += 1

        except EOFError:
            print("EOFError: stack_size %s, but can't find img no. %s" %(stack_size, i)) 
            break
    print ("split_stack finished.")

    return num_file




main()


