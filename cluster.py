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

thresh = 3
clusters = hcluster.fclusterdata(data2, thresh, criterion="distance")
print(clusters)

plt.scatter(*np.transpose(data), c=clusters,cmap="rainbow")
plt.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
plt.title(title)
plt.show()

