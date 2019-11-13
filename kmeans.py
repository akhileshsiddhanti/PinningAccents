import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.stats import describe
from scipy.spatial.distance import cdist


X = np.load("final_datasettop3.npy")[:,:-1]
true_labels = X[:,173]
X = X[:,:172]
clusterer = KMeans(n_clusters=9).fit(X)
clus = clusterer.fit_transform(X)
labels = clusterer.predict(X)
print(metrics.v_measure_score(true_labels,labels))
classes = ['Label 1', 'Label 2', 'Label 3']
ax = plt.scatter(clus[:, 0], clus[:, 1], c=clusterer.labels_)
plt.legend(handles=ax.legend_elements()[0], labels=classes)
plt.title('MFCC K-means')
plt.show()
