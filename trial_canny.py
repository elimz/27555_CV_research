'''
trials for: 
- canny edge detection on twin region;
- edge detection on mask file, to remove the top and bottom sections


'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("datasets/30_data/roi/twin_black.tif", 0)
height, width = img.shape[:2]


# --- this parts works but poorly === 
img = cv2.resize(img, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)
cv2.imshow("img", img)


edges = cv2.Canny(img, 60, 130)
edges = cv2.resize(edges, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)
img = cv2.resize(img, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC)
# show = np.hstack((edges, img))
# cv2.imshow("edgesedges_new", edges)
thresh_val = 30
_, edges_new  = cv2.threshold(edges, thresh_val, 255, cv2.THRESH_BINARY)
# cv2.imshow("edges_new", edges_new)

# -------- -------- -------- -------- 


mask = cv2.imread("datasets/30_data/mask_stack/img_12.tif")
# img
x_lo = 70
x_hi = 150
y_lo = 160
y_hi = 210
mask = mask[y_lo : y_hi, x_lo : x_hi]
# cv2.imshow("mask", mask)

# edges = cv2.Canny(img, 50, 150)  # apertureSize = 3

# kernel = np.ones((3,3), np.uint8)
# xfm = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
# xfm = cv2.cvtColor(xfm, cv2.COLOR_BGR2GRAY)
# cv2.circle( mask, (70, 160), 10, (255,0,0), -1)
# cv2.circle( mask, (150,210), 10, (0,255,0), -1)
y_lo_x = 60
y_hi_x  = 150
img = img[y_lo_x: y_hi_x, :]

lines = cv2.HoughLines(img,1,np.pi/180,200)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    


cv2.imshow('lines', img)
    

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



