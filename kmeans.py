import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.stats import describe
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dataset = np.load("mfcc_delta_data.npy")
print(dataset.shape)
X = dataset[:,:-1]
true_labels = dataset[:,-1]

pca = PCA(n_components=2)
X = pca.fit_transform(X)
print(X.shape)

clusterer = KMeans(n_clusters=3).fit(X)
clus = clusterer.fit_transform(X)
labels = clusterer.predict(X)

lda = LinearDiscriminantAnalysis(n_components=2)
lda = lda.fit(X,true_labels)
lda_transform = lda.fit_transform(X,true_labels)
print(lda.explained_variance_ratio_)
print(metrics.v_measure_score(true_labels,labels))
classes = ['Label 1', 'Label 2', 'Label 3']
ax = plt.scatter(lda_transform[:,0], lda_transform[:,1], c=true_labels)
plt.legend(handles=ax.legend_elements()[0], labels=list(set(true_labels)))
plt.title('MFCC K-means')
plt.show()
