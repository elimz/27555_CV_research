'''
trials for: 
- canny edge detection on twin region;
- edge detection on mask file, to remove the top and bottom sections


'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("datasets/30_data/roi/twin_black.tif", 0)
# img = cv2.imread("datasets/30_data/stack_img/img_12.tif", 0)


height, width = img.shape[:2]

img = cv2.resize(img, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)
# cv2.imshow("img", img)









# # ---------------- FAILED TRIAL ---------------- 
# # draw keypoints
# fast = cv2.FastFeatureDetector_create()
# kp = fast.detect(img, None)
# new = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
# new = cv2.resize(new, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)
# cv2.imshow("Fast Feature Detector - keypoints", new)

# fast_1 = cv2.FastFeatureDetector_create()
# kp_1 = fast_1.detect(out, None)
# new_1 = cv2.drawKeypoints(out, kp, None, color=(0,255,0))
# new_1 = cv2.resize(new_1, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)
# cv2.imshow("Fast Feature Detector - out", new_1)

# # ---------------- end keypoints ---------------- 

# # ---------------- Sharpening ---------------- 
# # trial for sharpening
# kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# # kernel_sharpen = np.array([[1,1,1], [1,-5, 1], [1,1,1]])
# out = cv2.filter2D(img, -1, kernel_sharpen)
#     # "-1" so output image will have the same depth as source image;
# cv2.imshow("out", out)
# # ---------------- end Sharpening ---------------- 


# # ---------------- Houghline ---------------- 
# lines = cv2.HoughLines(img,1,np.pi/180,200)

# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
    
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
  
# cv2.imshow('lines', img)
# # ---------------- end Houghline ----------------   


# # ---------------- USING: Canny ----------------   
# # --- this parts works but poorly === 
# img = cv2.resize(img, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)
# cv2.imshow("img", img)


# edges = cv2.Canny(img, 60, 130)
# edges = cv2.resize(edges, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)
# img = cv2.resize(img, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)
# # show = np.hstack((edges, img))
# # cv2.imshow("edgesedges_new", edges)
# thresh_val = 30
# _, edges_new  = cv2.threshold(edges, thresh_val, 255, cv2.THRESH_BINARY)
# # cv2.imshow("edges_new", edges_new)

# # -------- -------- -------- -------- 


# # edges = cv2.Canny(img, 50, 150)  # apertureSize = 3

# # trial for sharpening
def sharpening (img):

    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # kernel_sharpen = np.array([[1,1,1], [1,-5, 1], [1,1,1]])
    out = cv2.filter2D(img, -1, kernel_sharpen)
        # "-1" so output image will have the same depth as source image;
    # cv2.imshow("out", out),
    return out



kernel = np.ones((3,3), np.uint8)
img = cv2.equalizeHist(img)     # increases contrast of dark regions;
cv2.imshow("eqlz", img)

xfm = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# thresh_val = 240
# _, thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
cv2.imshow("xfm", xfm)
# trial: erosion, to remove bkg noise; 
erosion = cv2.erode(img, kernel, iterations = 1 )
# cv2.imshow("erosion", erosion)
dilate = cv2.dilate(erosion, kernel, iterations = 1 )
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
show = np.hstack((img, dilate))
cv2.imshow('hist eqlzn + erosion-dilation, kernel (3,3)', show)

# try morph_gradient?



# A *MUST* if use cv2.imshow to debug
while (1):

    k = cv2.waitKey(1) & 0xFF
    if (k == 27) or (k == ord("q")): 
        print "User_int: Quit key pressed."
        break
cv2.destroyAllWindows()

'''
Notes: 
- histogram eqlz makes it really noisy for canny;



'''



