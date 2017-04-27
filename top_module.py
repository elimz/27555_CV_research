
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
# import imgSeg
import manip 
import make_mask


def main():

    # reads in a stack of aligned images, as outputted by ImageJ
    stack_path = 'datasets/4000_results/apr4_4000_flip.tif'
    stack_size = 188           # number of images in that stack

    # separate raw image stack; and write to a stack of individual impages; 
    num_files = split_stack(stack_path, stack_size)

    i = 0

    # ----- Step 1: make masks for each image, to mark out twin boundaries -----
    # and save to mask_stack folder; 

    # call functions in other module for pair-wise comparison
    while (i < stack_size):

        # find original img, and make a mask;
        name = str(i)
        if (i < 10):
            name = "00" + name
        elif (10 <= i < 100):
            name = "0" + name

        curr_path = "datasets/4000_results/stack_img/img_" + str(name) + ".tif" 
        curr_res = make_mask.main(curr_path)

        res_name = "datasets/4000_results/mask_stack/img_" + str(name) + ".tif"
        # if (i < 10):
        #     res_name = 'datasets/4000_results/mask_stack/img_00%s.tif'%(i,)
        # elif (10 <= i < 100):
        #     res_name = 'datasets/4000_results/mask_stack/img_0%s.tif'%(i,)
        # else: 
        #     res_name = 'datasets/4000_results/mask_stack/img_%s.tif'%(i,)
            
        # res_name = "datasets/4000_results//img_" + str(i) + ".tif"
        cv2.imwrite(res_name, curr_res)
        i += 1
    print "++ step 2/3: finished writing all masks. "


    # ----- Step 2: call manip.py to do image segmentations on each image;  -----
    # outputs masks to manip_img folder
    k = 0
    while (k < stack_size):
        name_k = str(k)
        if (k < 10):
            name_k = "00" + name_k
        elif (10 <= k < 100):
            name_k = "0" + name_k

        # call manip on each image, and output to manip_img folder;
        img_path = "datasets/4000_results/stack_img/img_" + str(name_k) + ".tif" 
        mask_path = "datasets/4000_results/mask_stack/img_" + str(name_k) + ".tif"
        manip_res = manip.main(img_path, mask_path)

        # if (i < 10):
        #     res_name = 'datasets/4000_results/manip_stack_contours/img_00%s.tif'%(i,)
        # elif (10 <= i < 100):
        #     res_name = 'datasets/4000_results/manip_stack_contours/img_0%s.tif'%(i,)
        # else: 
        #     res_name = 'datasets/4000_results/manip_stack_contours/img_%s.tif'%(i,)
        res_name = "datasets/4000_results/manip_stack_contours/img_" + name_k + ".tif"
        # manip_name = "datasets/4000_results/manip_stack_contours/img_" + str(k) + ".tif"
        cv2.imwrite(res_name, manip_res)
        k += 1
    print "++ step 3/3: finished writing all manipulated images. "



    #######
    



    # # after done with all manip, 
    # # wait for keyboard interruption: Q key, or esc
    # key_int = cv2.waitKey(0)
    # if ((key_int == ord('q')) or (key_int == 27)):              
    #     print "User_int: Quit key pressed."
    #     cv2.destroyAllWindows()




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
            # save it "001" instead of "1"
            if (i < 10):
                name = 'datasets/4000_results/stack_img/img_00%s.tif'%(i,)
            elif (10 <= i < 100):
                name = 'datasets/4000_results/stack_img/img_0%s.tif'%(i,)
            else: 
                name = 'datasets/4000_results/stack_img/img_%s.tif'%(i,)
            
            stack.save(name)
            num_file += 1

        except EOFError:
            print("EOFError: stack_size %s, but can't find img no. %s" %(stack_size, i)) 
            break
    print ("++ step 1/3: split_stack finished.")

    return num_file




main()
sys.exit()

