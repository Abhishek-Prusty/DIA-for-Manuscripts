import argparse
import json
import os
import pickle as pkl 
import cv2
import numpy as np
import colorsys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--page", required=True,
                help="Path to the image")
ap.add_argument("-m", "--mask", required=True,
                help="Path to the mask")

args = vars(ap.parse_args())
imgs=args["page"]
imgs2=args["mask"]
img=cv2.imread(imgs,1)
img2=cv2.imread(imgs,0)

with open('segments.pkl','rb') as f:
    segments=pkl.load(f)

#print(segments[0])
segments=sorted(segments,key=lambda x:(x[1]))
segments=sorted(segments,key=lambda x:(x[1]*x[0]))

#print(segments[0])
orb = cv2.ORB_create(edgeThreshold=5,nfeatures=50, scoreType=cv2.ORB_HARRIS_SCORE) 
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
print(len(segments))
no_segs=len(segments)
temp_inc=0
matt_inc=0
count=0
for segment in segments:
    img=cv2.imread(imgs,1)
    img2=cv2.imread(imgs,0)
    #template=img2[(1-int(temp_inc/100))*segment[1]:(1+int(temp_inc/100))*segment[3],(1-int(temp_inc/100))*segment[0]:(1+int(temp_inc/100))*segment[2]]
    template=img2[segment[1]-temp_inc:segment[3]+temp_inc,segment[0]-temp_inc:segment[2]+temp_inc]
    
    w, h = template.shape[::-1]
    kp1 ,des1= orb.detectAndCompute(template,None)
    cv2.imshow("template",template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #t1=cv2.drawKeypoints(template,kp1,None)
    #cv2.imshow("t1",t1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if des1 is None :
        continue
    matches=list()
    for seg in segments:
        #matt=img2[(1-int(matt_inc/100))*seg[1]:(1+int(matt_inc/100))*seg[3],(1-int(matt_inc/100))*seg[0]:(1+int(matt_inc/100))*seg[2]]
        matt=img2[seg[1]-matt_inc:seg[3]+matt_inc,seg[0]-matt_inc:seg[2]+matt_inc]
        #cv2.imshow("matt",matt)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        w_m,h_m=matt.shape[::-1]
        kp2 ,des2= orb.detectAndCompute(matt,None)
        #t2=cv2.drawKeypoints(matt,kp2,None)
        #cv2.imshow("t2",t2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if des2 is None:
            continue
        matches = bf.knnMatch(np.asarray(des1, np.uint8), np.asarray(des2, np.uint8), k=2)
        #print(matches)

        if(len(matches)>30):
            cv2.rectangle(img, (seg[0],seg[1]), ( seg[2] , seg[3]), (0,0,255), 2)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", (int(img.shape[1]), int(img.shape[0])))
    cv2.imshow("output",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    count=count+1
    print(count)


    



