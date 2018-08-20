import cv2
import numpy as np

img = cv2.imread('IMG_0256_srimahaganapati_working.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(edgeThreshold=5,nfeatures=10000, scoreType=cv2.ORB_HARRIS_SCORE,scaleFactor=1.2) 
kp ,des= orb.detectAndCompute(gray,None)

img=cv2.drawKeypoints(gray,kp,None)

cv2.imwrite('sift_keypoints.jpg',img)
