#importing the libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the Mall data set
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#using the elbow method to find the number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#applying k means to mall data set
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=0)
ykmeans = kmeans.fit_predict(X)

#visualizing the clusterer
plt.scatter(X[ykmeans==0,0], X[ykmeans == 0, 1],s =100, c='red', label = 'Careful')
plt.scatter(X[ykmeans==1,0], X[ykmeans == 1, 1],s =100, c='blue', label = 'Standard')
plt.scatter(X[ykmeans==2,0], X[ykmeans == 2, 1],s =100, c='green', label = 'Target')
plt.scatter(X[ykmeans==3,0], X[ykmeans == 3, 1],s =100, c='cyan', label = 'Careless')
plt.scatter(X[ykmeans==4,0], X[ykmeans == 4, 1],s =100, c='magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s =300, c = 'yellow', label = 'Centroids')
plt.title('Clustering of the client')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1 - 100)')
plt.legend()
plt.show()
   
