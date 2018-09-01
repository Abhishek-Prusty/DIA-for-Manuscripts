import argparse
import json
import os
import pickle as pkl 
import cv2
import numpy as np
import colorsys
from sklearn.metrics import jaccard_similarity_score
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt 

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


mxw=-999999999999
mxh=-999999999999

for i in range(len(segments)):
    mxw=max(mxw,segments[i][2]-segments[i][0])
    mxh=max(mxh,segments[i][3]-segments[i][1])

#print(mxw,"   ",mxh)



#print(segments[0])
segments=sorted(segments,key=lambda x:(x[1]))
segments=sorted(segments,key=lambda x:(x[1]*x[0]))

#print(segments[0])

#print(len(segments))
no_segs=len(segments)
segments=segments[13:]
#parameters
temp_inc=1
matt_inc=1
Bound_width=5
Bound_height=5
threshold=0.5
scale_low=97
scale_high=103
rotate_low=-3
rotate_high=3
count=0



matt_aug=[]
count1=0
print("precomputing augmented components")
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

def jacc(matt,template):
    res=jaccard_similarity_score(matt,template)
    return res

def skel(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
     
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
     
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel

print("template matching")
#print(matt_aug)
for segment in segments:

    img=cv2.imread(imgs,1)
    img2=cv2.imread(imgs2,0)
    #template=img2[(1-int(temp_inc/100))*segment[1]:(1+int(temp_inc/100))*segment[3],(1-int(temp_inc/100))*segment[0]:(1+int(temp_inc/100))*segment[2]]
    template=img2[segment[1]-temp_inc:segment[3]+temp_inc,segment[0]-temp_inc:segment[2]+temp_inc]
    template=skel(template)
    template=cv2.resize(template,(mxw,mxh))
    
    #val,template = cv2.threshold(template,100,255,cv2.THRESH_BINARY)
    cv2.imshow("template",template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    count=0

    for seg in segments:
        #matt=img2[(1-int(matt_inc/100))*seg[1]:(1+int(matt_inc/100))*seg[3],(1-int(matt_inc/100))*seg[0]:(1+int(matt_inc/100))*seg[2]]
        matt=img2[seg[1]-matt_inc:seg[3]+matt_inc,seg[0]-matt_inc:seg[2]+matt_inc]
        matt=cv2.resize(matt,(mxw,mxh))
        for matt1 in matt_aug[count]:

            #cv2.imshow("matt1",matt1)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #print("1")
            matt1=cv2.resize(matt1,(mxw,mxh))
            #val,matt1 = cv2.threshold(matt1,100,255,cv2.THRESH_BINARY)

            template=cv2.resize(template,(mxw,mxh))
            res = cv2.matchTemplate(matt1,template,cv2.TM_CCOEFF_NORMED)
            if(max(res.flatten())>threshold):
                cv2.rectangle(img, (seg[0],seg[1]),( seg[2] , seg[3]), (0,0,255), 2)

            '''
            res=jaccard_similarity_score(matt1.flatten(),template.flatten())
            #print(res)
            if(res>0.5):
                cv2.rectangle(img, (seg[0],seg[1]),( seg[2] , seg[3]), (0,0,255), 2)
            '''
        count=count+1
    

    #plt.show()
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", (int(img.shape[1]), int(img.shape[0])))
    cv2.imshow("output",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    



