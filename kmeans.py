import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


X = np.load("outfilemfcctop9.npy")
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
clusterer = KMeans(n_clusters=9).fit(X)
clus = clusterer.fit_transform(X)
classes = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label 6', 'Label 7', 'Label 8', 'Label 9']
ax = plt.scatter(clus[:, 0], clus[:, 1], c=clusterer.labels_)
plt.legend(handles=ax.legend_elements()[0], labels=classes)
plt.title('MFCC K-means')
plt.show()
