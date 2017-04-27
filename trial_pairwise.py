import cv2
import numpy as np

# read in 2 images;
# stack_img = cv2.imread('datasets/4000_results/stack_img/img_13.tif', 0)
# stack_img2 = cv2.imread('datasets/4000_results/stack_img/img_14.tif', 0)

# manip_img = cv2.imread('datasets/4000_results/manip_stack_contours/img_13.tif', 0)


# assert (stack_img.shape == manip_img.shape),("Error: stack_img and manip_img diff size.")
# (height, width) = ( stack_img.shape)[0:2]

# want to output a folder of kp results; 


# if keypoints are consistent, use keypoints to overlay 2 images; 
# then somehow pixel comparison; since grain features are more likely to line up 
    # pair-wise comparison
num = 20
# repath = "datasets/30_data/kp_stack/"
detector = cv2.SimpleBlobDetector_create()

for i in range (0, num-1):
    im = cv2.imread("datasets/30_data/stack_img/img_" + str(i) + ".tif")
    # find key points and draw;
    kp = detector.detect(im)
    res = cv2.drawKeypoints(im, kp, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # save 
    res_name = "datasets/30_data/kp_stack/img_" + str(i) + ".tif"
    cv2.imwrite(res_name, res)


print "done"



# TRIAL: KEYPOINTS
# get a list of keypoints 

# res = stack_img.copy()

# # blob 
# detector = cv2.SimpleBlobDetector_create()
# kp = detector.detect(stack_img)
# draw = cv2.drawKeypoints(stack_img, kp, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# draw = cv2.resize(draw, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
# cv2.imshow('res', draw)

# detector2 = cv2.SimpleBlobDetector_create()
# kp2 = detector.detect(stack_img2)
# draw2 = cv2.drawKeypoints(stack_img2, kp, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# draw2 = cv2.resize(draw2, (height / 2, width / 2), interpolation = cv2.INTER_CUBIC) 
# cv2.imshow('res2', draw2)


cv2.waitKey(0)
cv2.destroyAllWindows()





# # failed attempt: compare pixel -by pixel tells you everything's wrong
# what about straightup compairiosn
# max_cnt = 10
# cnt = 0
# cv2.circle(res, (300,100), 10, (255, 255, 255), 3)

# for row in range(200, height/2 -1):
#     for col in range(0, width - 1):
#         if (stack_img[row][col] != manip_img[row][col]):
#             # found difference, draw a dot;
#             if (cnt <= max_cnt):
#                 cv2.circle(res, (col, row), 10, (255, 0, 255), 3)
#             cnt += 1

# cv2.imshow('res', res)

# while(1):
#     k = cv2.waitKey(1) & 0xFF
#     if (k == 27) or (k == ord('q')):
#         print "user_int, quit key pressed."
#         break
# cv2.destroyAllWindows()




