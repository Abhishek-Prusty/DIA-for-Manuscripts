import numpy as np 
import pickle as pkl 
import cv2
import argparse
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
import pandas as pd 


with open('segments.pkl','rb') as f:
    segments=pkl.load(f)

data=[]
for i in range(len(segments)):
	data.append(segments[i][:2])
data2=[]
for i in range(len(segments)):
	data2.append(segments[i][1:2])

data=np.array(data)
data2=np.array(data2)

thresh = 20
clusters = hcluster.fclusterdata(data2, thresh, criterion="distance")
print((clusters))

print(len(data2))
no_clusters=len(set(clusters))
lines=[]


plt.scatter(*np.transpose(data), c=clusters,cmap="rainbow")
#plt.scatter()
plt.axis("equal")
title = "thresh: %f, no. clu: %d" % (thresh, no_clusters)
plt.title(title)
plt.show()

