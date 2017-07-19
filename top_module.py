
# --------------------------------------------------
# Experimenting Image Segmentation on 3D dataset 
#               with Python OpenCV 3.0
# 
# Prof. Degraef
# Elim Zhang, Version 2
#   yilinz@andrew
# Dec. 25, 2016 (last edit July, 2017)
# --------------------------------------------------
# This is the Top Module for this project;
#   - It first reads in a multi-page tiff file, an aligned stack outputted by Image J;
#   - Adjusts image contrast using histogram equalization;
#   - Calls 'make_mask.py' to produce a stack of masks, which are used to separate 
#       twin and grain regions, which have different contrasts and thus require 
#       different segmentation methods;
#   - Calls 'manip.py' to run segmentation on the two regions of each slice in stack;
#       returns the segmented results 
# --------------------------------------------------

import cv2 
from PIL import Image
import sys

# import subsidiary files: 
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

    while (i < stack_size):

        # find original img, and make a mask;
        name = str(i)
        if (i < 10):
            name = "00" + name
        elif (10 <= i < 100):
            name = "0" + name

        curr_path = "datasets/4000_results/stack_img/img_" + str(name) + ".tif" 
        curr_mask = make_mask.main(curr_path)

        mask_name = "datasets/4000_results/mask_stack/img_" + str(name) + ".tif"

        cv2.imwrite(mask_name, curr_mask)
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
        res_name = "datasets/4000_results/manip_stack_contours/img_" + name_k + ".tif"
        cv2.imwrite(res_name, manip_res)
        k += 1

    print "++ step 3/3: finished writing all manipulated images. "

    # ----- Step (3): call manip.py to do image segmentations on each image;  -----
    # temporary: for the final presentation; 
    #       extract an ROI to form a stack to visualize 3D grain structure;
    j = 0 
    while(j < stack_size):
        name_j = str(j)
        if (j < 10):
            name_j = "00" + name_j
        elif (10 <= j < 100):
            name_j = "0"+name_j
        img_path =  "datasets/4000_results/manip_stack_contours/img_" + name_j + ".tif"
        img = cv2.imread(img_path)
        
        y_lo_1, y_hi_1 = 100, 600
        x_lo_1, x_hi_1 = 100, 600

        roi = img[y_lo_1 : y_hi_1, x_lo_1:x_hi_1]
        roi_name = "datasets/4000_results/roi_stack/img_"+ name_j + ".tif"

        cv2.imwrite(roi_name, roi)
        j += 1

    print  "++ temp step: finished writing all manipulated images. "


# split multi-page tiff image (output from ImageJ after stack alignment) into 
#   individual images; (modified code from Stackoverflow)
# returns the number of files read; 
# save all images with name "img_<number>.tif", under folder stack_img in 
#   current directory. 
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

