import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


X = np.load("outfilemfcc.npy")
X = np.transpose(X)
distortions = []
def elbow_method():
    for n_clusters in range(1,21):
        clusterer = KMeans(n_clusters=n_clusters).fit(X)
        clusterer.fit(X)
        distortions.append(sum(np.min(cdist(X, clusterer.cluster_centers_,'euclidean'),axis=1)) / X.shape[0])
    plt.plot(range(1,21), distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()

#optimal clusters are 7
clusterer = KMeans(n_clusters=7).fit(X)
clus = clusterer.fit(X)
plt.scatter(clus[:, 0], clus[:, 1], c=cluster.labels_)
plt.show()
