import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

ENG_SIZE = 150

# Load the dataset
dataset = np.load('final_dataset_top3_delta.npy')
X = dataset[:,:-1]
y = dataset[:,-1]

print(np.unique(y,return_counts=True))

few_english = np.random.permutation(np.argwhere(y==1))[:ENG_SIZE]

Xf = np.vstack((X[few_english][:,0,:], X[y!=1]))
yf = np.hstack((y[few_english][:,0], y[y!=1]))

x, xt, y, yt = train_test_split(Xf,yf,test_size=0.20,random_state=42,stratify=yf,shuffle=True)

# SVC
from sklearn.svm import SVC

svc = SVC().fit(x,y)
print("SVC Train:"+str(svc.score(x,y)))
print("SVC Test:"+str(svc.score(xt,yt)))
print("SVC Confusion Matrix:")
print(confusion_matrix(yt,svc.predict(xt)))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(x,y)
print("LogisticRegression Train:"+str(lr.score(x,y)))
print("LogisticRegression Test:"+str(lr.score(xt,yt)))
print("LogisticRegression Confusion Matrix:")
print(confusion_matrix(yt,lr.predict(xt)))

# K Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15).fit(x,y)
print("KNN Train:"+str(knn.score(x,y)))
print("KNN Test:"+str(knn.score(xt,yt)))
print("KNN Confusion Matrix:")
print(confusion_matrix(yt,knn.predict(xt)))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(x,y)
print("GNB Train:"+str(gnb.score(x,y)))
print("GNB Test:"+str(gnb.score(xt,yt)))
print("GNB Confusion Matrix:")
print(confusion_matrix(yt,gnb.predict(xt)))

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier().fit(x,y)
print("DTC Train:"+str(dtc.score(x,y)))
print("DTC Test:"+str(dtc.score(xt,yt)))
print("DTC Confusion Matrix:")
print(confusion_matrix(yt,dtc.predict(xt)))

# Random Forests
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier().fit(x,y)
print("RandomForestClassifier Train:"+str(rfc.score(x,y)))
print("RandomForestClassifier Test:"+str(rfc.score(xt,yt)))
print("Random Confusion Matrix:")
print(confusion_matrix(yt,rfc.predict(xt)))