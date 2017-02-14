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



def histogram(path1, path2):
    print("path = %s" %path1)
    img_1 = cv2.imread(path1, 0)
    img_2 = cv2.imread(path2, 0)

    hist_full_1 = cv2.calcHist([img_1],[0],None,[256],[0,256])
    hist_full_2 = cv2.calcHist([img_2],[0],None,[256],[0,256])

    # calcualte histogram
    
    plt.subplot(221), plt.imshow(img_1, 'gray')
    plt.title("img_1")
    
    plt.subplot(222), plt.imshow(img_2,'gray')
    plt.title("img_2")
    
    plt.subplot(223), plt.plot(hist_full_1)
    plt.title("img_1")
    plt.ylabel("Pixel Values")
    plt.xlabel("No. of Pixels")

    plt.subplot(224), plt.plot(hist_full_2)
    plt.title("img_2")
    plt.ylabel("Pixel Values")
    plt.xlabel("No. of Pixels")
    plt.show()




histogram("datasets/30_data/stack_img/img_0.tif", 
          "datasets/30_data/stack_img/img_1.tif")







