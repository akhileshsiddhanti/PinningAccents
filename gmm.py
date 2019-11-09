import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


X = np.load("outfilemfcc.npy")
X = np.transpose(X)
n_components = np.arange(1,10)
BIC = np.zeros(n_components.shape)
AIC = np.zeros(n_components.shape)
def aicbic():
    for i,ncomp in enumerate(n_components):
        gmm = GaussianMixture(n_components=ncomp).fit(X)
        AIC[i] = gmm.aic(X)
        BIC[i] = gmm.bic(X)

    plt.figure()
    plt.plot(n_components, AIC, label='AIC')
    plt.plot(n_components, BIC, label='BIC')
    plt.legend(loc=0)
    plt.xlabel('n_components')
    plt.ylabel('AIC / BIC')
    plt.title('Information criterion for GMM')
    plt.show()

#optimal is 4 n_components
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
classes = ['Label 1', 'Label 2','Label 3', 'Label 4']
ax = plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.legend(handles=ax.legend_elements()[0], labels=classes)
plt.title('MFCC GaussianMixture')
plt.show()
