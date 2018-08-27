import numpy as np 
import pickle as pkl 
import cv2
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
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

cluster = AgglomerativeClustering(n_clusters=100, affinity='manhattan', linkage='average')  
cluster.fit_predict(data2)
print(cluster.labels_)
plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow') 
plt.show()

