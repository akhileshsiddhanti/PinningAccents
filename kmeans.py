import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics


X = np.load("mfcc_dataset.npy")
true_labels = X[:,21]
X = X[:,:20]

clusterer = KMeans(n_clusters=9).fit(X)
clus = clusterer.fit_transform(X)
labels = clusterer.predict(X)
print(metrics.v_measure_score(true_labels,labels))
classes = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label 6', 'Label 7', 'Label 8', 'Label 9']
ax = plt.scatter(clus[:, 0], clus[:, 1], c=clusterer.labels_)
plt.legend(handles=ax.legend_elements()[0], labels=classes)
plt.title('MFCC K-means')
plt.show()
