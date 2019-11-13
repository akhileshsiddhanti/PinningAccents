import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn import metrics

X = np.load("final_datasettop3.npy")[:,:-1]
true_labels = X[:,173]
X = X[:,:172]

gmm = GaussianMixture(n_components=9).fit(X)
labels = gmm.predict(X)
print(metrics.v_measure_score(true_labels,labels))
classes = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label 6', 'Label 7', 'Label 8', 'Label 9']
ax = plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.legend(handles=ax.legend_elements()[0], labels=classes)
plt.title('MFCC GaussianMixture')
plt.show()
