import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ENG_SIZE = 150
np.random.seed(42)

# Load the dataset
dataset = np.load('final_dataset_top3_delta.npy')
X = dataset[:,:-1]
y = dataset[:,-1]

few_english = np.random.permutation(np.argwhere(y==1))[:ENG_SIZE]

Xf = np.vstack((X[few_english][:,0,:], X[y!=1]))
yf = np.hstack((y[few_english][:,0], y[y!=1]))

x, xt, y, yt = train_test_split(Xf,yf,test_size=0.20,random_state=42,stratify=yf,shuffle=True)

# # SVC
# from sklearn.svm import SVC

# parameters = {'C':[0.0001,0.001,0.01,0.1,1,10,100]}
# svc = SVC(kernel='rbf')
# clf = GridSearchCV(svc, parameters,cv=10,	 n_jobs=7)
# clf.fit(x,y)
# print("SVC Train:"+str(clf.score(x,y)))
# print("SVC Test:"+str(clf.score(xt,yt)))
# print("SVC Confusion Matrix:")
# print(confusion_matrix(yt,clf.predict(xt)))
# print(clf.best_score_)
# print(clf.best_params_)
# print(svc.coef_)

# # Logistic Regression
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression().fit(x,y)
# print("LogisticRegression Train:"+str(lr.score(x,y)))
# print("LogisticRegression Test:"+str(lr.score(xt,yt)))
# print("LogisticRegression Confusion Matrix:")
# print(confusion_matrix(yt,lr.predict(xt)))
# print(lr.coef_)

# # K Nearest Neighbours
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=15).fit(x,y)
# print("KNN Train:"+str(knn.score(x,y)))
# print("KNN Test:"+str(knn.score(xt,yt)))
# print("KNN Confusion Matrix:")
# print(confusion_matrix(yt,knn.predict(xt)))

# # Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB().fit(x,y)
# print("GNB Train:"+str(gnb.score(x,y)))
# print("GNB Test:"+str(gnb.score(xt,yt)))
# print("GNB Confusion Matrix:")
# print(confusion_matrix(yt,gnb.predict(xt)))

# # Decision Trees
# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier().fit(x,y)
# print("DTC Train:"+str(dtc.score(x,y)))
# print("DTC Test:"+str(dtc.score(xt,yt)))
# print("DTC Confusion Matrix:")
# print(confusion_matrix(yt,dtc.predict(xt)))
# print(np.argsort(dtc.feature_importances_)[::-1][:5])

# # Random Forests
from sklearn.ensemble import RandomForestClassifier

parameters = {'n_estimators':[3,5,7,10,13,15,20,25], 'max_depth':[None,10,20,30,40,50]}
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc, parameters,cv=10, n_jobs=7)
clf.fit(x,y)
print("RandomForestClassifier Train:"+str(clf.score(x,y)))
print("RandomForestClassifier Test:"+str(clf.score(xt,yt)))
print("Random Confusion Matrix:")
print(confusion_matrix(yt,clf.predict(xt)))
print(clf.best_params_)
print(clf.best_score_)
print(np.argsort(rfc.feature_importances_)[::-1][:5])