import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dataset = np.load("final_dataset_top3_delta.npy")
X = dataset[:,:-1]
true_labels = dataset[:,-1]

gmm = GaussianMixture(n_components=3).fit(X)
labels = gmm.predict(X)
lda = LinearDiscriminantAnalysis(n_components=2)
lda = lda.fit(X,true_labels)
lda_transform = lda.fit_transform(X,true_labels)
print(lda.explained_variance_ratio_)
print(metrics.v_measure_score(true_labels,labels))
classes = ['Label 1', 'Label 2', 'Label 3']
ax = plt.scatter(lda_transform[:,0], lda_transform[:,1], c=labels)
plt.legend(handles=ax.legend_elements()[0], labels=classes)
plt.title('MFCC GaussianMixture')
plt.show()
