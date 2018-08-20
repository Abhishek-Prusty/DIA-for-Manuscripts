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

print(len(segments))
no_segs=len(segments)

#parameters
temp_inc=0
matt_inc=2
Bound_width=7
Bound_height=7
threshold=0.7

count=0
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
    for seg in segments:
        #matt=img2[(1-int(matt_inc/100))*seg[1]:(1+int(matt_inc/100))*seg[3],(1-int(matt_inc/100))*seg[0]:(1+int(matt_inc/100))*seg[2]]
        matt=img2[seg[1]-matt_inc:seg[3]+matt_inc,seg[0]-matt_inc:seg[2]+matt_inc]
        #cv2.imshow("matt",matt)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        w_m,h_m=matt.shape[::-1]
        print(w," ",h," ")
        print(w_m," ",h_m)
        if(w+Bound_width>w_m>=w and h+Bound_height>h_m>=h):
            res = cv2.matchTemplate(matt,template,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            mx=max(res.flatten())
            loc = np.where( res == mx)
            print("loc",loc)
            if(max(res.flatten())>threshold):
                cv2.rectangle(img, (seg[0],seg[1]), ( seg[2] , seg[3]), (0,0,255), 2)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", (int(img.shape[1]), int(img.shape[0])))
    cv2.imshow("output",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    count=count+1
    print(count)


    



