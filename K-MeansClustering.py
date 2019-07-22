#K-Means Clustering 

#Importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#Importing dataset 
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

#Using elbow method to find optimal nunmber of clusters 
from sklearn.cluster import KMeans 
wcss = []
for i in range(1,11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show( )

#Fitting K means method to the dataset 
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
Y_kmeans = kmeans.fit_predict(X)

#Vizualizing Test Results 
plt.scatter(X[:,0], X[:,1], s = 50, c = 'black')
plt.title("Before Clustering")
plt.xlabel('Annual Income($)')
plt.ylabel("Spending Score(1-100)")
plt.show 

plt.scatter(X[Y_kmeans == 0, 0], X[Y_kmeans == 0,1], s = 50, c = 'red', label = 'Careful') 
plt.scatter(X[Y_kmeans == 1, 0], X[Y_kmeans == 1,1], s = 50, c = 'blue', label = 'Moderate') 
plt.scatter(X[Y_kmeans == 2, 0], X[Y_kmeans == 2,1], s = 50, c = 'magenta', label = 'Target') 
plt.scatter(X[Y_kmeans == 3, 0], X[Y_kmeans == 3,1], s = 50, c = 'green', label = 'Careless') 
plt.scatter(X[Y_kmeans == 4, 0], X[Y_kmeans == 4,1], s = 50, c = 'black', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title("Clusters of clients")
plt.xlabel('Annual Income($)')
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show 

    
                            

