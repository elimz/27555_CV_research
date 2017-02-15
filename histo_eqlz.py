# 27555 Computer Vision for Microstructure 3D visualization;
# Elim Zhang, Prof. Degraef's lab
# Written Feb. 14, 2017
#
# part of helper functions, called by topMod
#
# Histogram Equalization:
# increasing image contrasts, before image segmentation;
# Code followed examples from OpenCV documentations on histogram Equalization;
# Elim Zhang,

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# def histogram(path):
#     img = cv2.imread(path, 0)

#     hist,bins = np.histogram(img.flatten(), 256, [0,256])
#     cdf = hist.cumsum()
#     cdf_normalized = cdf * hist.max() / cdf.max()

#     plt.plot(cdf_normalized, color = 'b')
#     plt.hist(img.flatten(),256,[0,256], color = 'r')
#     plt.xlim([0,256])
#     plt.legend(('cdf','histogram'), loc = 'upper left')
#     plt.show()



''' 
using opencv instead
'''
def histogram (path):
    img = cv2.imread(path,0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ))
    

    # scaling
    width, height = res.shape[:2]   # take first 2 componenets of tuple for h, w

    res = cv2.resize(res, 
                    (height / 2, width / 2), 
                    interpolation = cv2.INTER_CUBIC)  
    cv2.imshow("result after histogram equalization", res)


    # # after done with all manip, 
    # # wait for keyboard interruption: Q key, or esc
    # key_int = cv2.waitKey(0)
    # if ((key_int == ord('q')) or (key_int == 27)):              
    #     print "User_int: Quit key pressed."
    #     cv2.destroyAllWindows()


# histogram("datasets/30_data/stack_img/img_10.tif")



''' 
following section takes in and display two paths
''' 
# def histogram(path1, path2):
#     print("path = %s" %path1)
#     img_1 = cv2.imread(path1, 0)
#     img_2 = cv2.imread(path2, 0)

#     hist_full_1 = cv2.calcHist([img_1],[0],None,[256],[0,256])
#     hist_full_2 = cv2.calcHist([img_2],[0],None,[256],[0,256])

#     # calcualte histogram
    
#     plt.subplot(221), plt.imshow(img_1, 'gray')
#     plt.title("img_1")
    
#     plt.subplot(222), plt.imshow(img_2,'gray')
#     plt.title("img_2")

#     plt.subplot(223), plt.plot(hist_full_1)
#     plt.title("img_1")
#     plt.ylabel("Pixel Values")
#     plt.xlabel("No. of Pixels")

#     plt.subplot(224), plt.plot(hist_full_2)
#     plt.title("img_2")
#     plt.ylabel("Pixel Values")
#     plt.xlabel("No. of Pixels")
#     plt.show()




# histogram("datasets/30_data/stack_img/img_0.tif", 
#           "datasets/30_data/stack_img/img_1.tif")







