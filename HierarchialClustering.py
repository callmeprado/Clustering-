#Importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#Importing dataset 
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

#Using Dendograms to find total number of clusters 
plt.figure(figsize = (10,8))

from scipy.cluster.hierarchy import dendrogram, linkage 
dendrogram = dendrogram(linkage(X, method = 'ward')) 
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidian Distances")
plt.show()

#Fitting HC to the dataset (aggolomerative)
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
Y_link = hc.fit_predict(X)

#Vizualizing the Test Results 
plt.scatter(X[:,0], X[:,1], s = 50, c = 'black')
plt.title("Before Clustering")
plt.xlabel('Annual Income($)')
plt.ylabel("Spending Score(1-100)")
plt.show 

plt.scatter(X[Y_link == 0, 0], X[Y_link == 0,1], s = 50, c = 'red', label = 'Careful') 
plt.scatter(X[Y_link == 1, 0], X[Y_link == 1,1], s = 50, c = 'blue', label = 'Moderate') 
plt.scatter(X[Y_link == 2, 0], X[Y_link == 2,1], s = 50, c = 'magenta', label = 'Target') 
plt.scatter(X[Y_link == 3, 0], X[Y_link == 3,1], s = 50, c = 'green', label = 'Careless') 
plt.scatter(X[Y_link == 4, 0], X[Y_link == 4,1], s = 50, c = 'black', label = 'Sensible')
plt.title("Clusters of clients")
plt.xlabel('Annual Income($)')
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show 