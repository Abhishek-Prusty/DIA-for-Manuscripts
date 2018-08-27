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
img2=cv2.imread(imgs2,0)

with open('segments.pkl','rb') as f:
    segments=pkl.load(f)

#print(segments[0])
segments=sorted(segments,key=lambda x:(x[1]))
segments=sorted(segments,key=lambda x:(x[1]*x[0]))

#print(segments[0])

#print(len(segments))
no_segs=len(segments)
segments=segments[15:]
#parameters
temp_inc=0
matt_inc=2
Bound_width=7
Bound_height=7
threshold=0.55
scale_low=90
scale_high=110
rotate_low=-5
rotate_high=+5
count=0

matt_aug=[]
count1=0
for segment in segments:
    img2=cv2.imread(imgs2,0)
    matt2=img2[segment[1]-matt_inc:segment[3]+matt_inc,segment[0]-matt_inc:segment[2]+matt_inc]
    matt_aug.append([])
    for i in range(scale_low,scale_high+1,1):
            try:
                res = cv2.resize(matt2,None,fx=float(i/100), fy=float(i/100), interpolation = cv2.INTER_CUBIC)
                matt_aug[count1].append(res)
            except:
                continue
    
    for i in range(rotate_low,rotate_high+1,1):
            try:
                M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),i,1)
                dst = cv2.warpAffine(matt2,M,(cols,rows))
                matt_aug[count1].append(dst)
            except:
                continue
    count1=count1+1
    
#print(matt_aug)
for segment in segments:
    img=cv2.imread(imgs,1)
    img2=cv2.imread(imgs2,0)
    #template=img2[(1-int(temp_inc/100))*segment[1]:(1+int(temp_inc/100))*segment[3],(1-int(temp_inc/100))*segment[0]:(1+int(temp_inc/100))*segment[2]]
    template=img2[segment[1]-temp_inc:segment[3]+temp_inc,segment[0]-temp_inc:segment[2]+temp_inc]
    cv2.imshow("template",template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    w, h = template.shape[::-1]
    matches=list()
    count=0
    for seg in segments:
        #matt=img2[(1-int(matt_inc/100))*seg[1]:(1+int(matt_inc/100))*seg[3],(1-int(matt_inc/100))*seg[0]:(1+int(matt_inc/100))*seg[2]]
        matt=img2[seg[1]-matt_inc:seg[3]+matt_inc,seg[0]-matt_inc:seg[2]+matt_inc]
        #cv2.imshow("matt",matt)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        w_m,h_m=matt.shape[::-1]
        #templat=cv2.resize(template,(w_m,h_m))
        cols=w_m
        rows=h_m

        for matt1 in matt_aug[count]:
            if(w+Bound_width>w_m>=w and h+Bound_height>h_m>=h):
                try:
                    res = cv2.matchTemplate(matt1,template,cv2.TM_CCOEFF_NORMED)
                    if(max(res.flatten())>threshold):
                        cv2.rectangle(img, (seg[0],seg[1]),( seg[2] , seg[3]), (0,0,255), 2)

                except:
                        try:
                            res = cv2.matchTemplate(template,matt1,cv2.TM_CCOEFF_NORMED)
                            if(max(res.flatten())>threshold):
                                cv2.rectangle(img, (seg[0],seg[1]), ( seg[2] , seg[3]), (0,0,255), 2)
                        except:
                            continue
        count=count+1

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", (int(img.shape[1]), int(img.shape[0])))
    cv2.imshow("output",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(count)


    



