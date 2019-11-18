import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.plotting import plot_decision_regions
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ENG_SIZE = 150
np.random.seed(42)

# Load the dataset
dataset = np.load('final_datasettop3.npy')
X = dataset[:,:-1]
y = dataset[:,-1]

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])

few_english = np.random.permutation(np.argwhere(y==1))[:ENG_SIZE]

Xf = np.vstack((X[few_english][:,0,:], X[y!=1]))
yf = np.hstack((y[few_english][:,0], y[y!=1]))

x, xt, y, yt = train_test_split(Xf,yf,test_size=0.20,random_state=42,stratify=yf,shuffle=True)

# from sklearn.svm import SVC
#
# parameters = {'C':[0.0001,0.001,0.01,0.1,1,10,100], 'gamma':[1e-3, 1e-4]}
# C = [0.0001,0.001,0.01,0.1,1,10,100]
# Gammas = [1e-3, 1e-4]
# svc = SVC(kernel='rbf')
# clf = GridSearchCV(svc, parameters,cv=10,n_jobs=7)
# clf.fit(x,y)
# print("SVC Train:"+str(clf.score(x,y)))
# print("SVC Test:"+str(clf.score(xt,yt)))
# print("SVC Confusion Matrix:")
# print(confusion_matrix(yt,clf.predict(xt)))
# cm = confusion_matrix(yt,clf.predict(xt))
# plt.subplot()
# sn.heatmap(cm, annot=True,fmt='g',cmap='Greens',square=True)
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t)
# plt.show()
# scores = clf.cv_results_['mean_test_score']
# scores = np.array(scores).reshape(len(C), len(Gammas))
# for ind, i in enumerate(C):
#     plt.plot(Gammas, scores[ind], label='C: ' + str(i))
# plt.legend()
# plt.xlabel('Gamma')
# plt.ylabel('Mean score')
# plt.title("Score of SVC parameters")
# plt.show()
# print(clf.best_score_)
# print(clf.best_params_)
# print(svc.coef_)
# fig = plt.figure(figsize=(10, 8))

# fig = plot_decision_regions(X=xt.astype(np.integer), y=yt.astype(np.integer),clf=clf, legend=2)
# plt.show()


# Logistic Regression
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression().fit(x,y)
# print("LogisticRegression Train:"+str(lr.score(x,y)))
# print("LogisticRegression Test:"+str(lr.score(xt,yt)))
# print("LogisticRegression Confusion Matrix:")
# print(confusion_matrix(yt,lr.predict(xt)))
# cm = confusion_matrix(yt,lr.predict(xt))
# plt.subplot()
# sn.heatmap(cm, annot=True,fmt='g',cmap='Greens',square=True)
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t)
# plt.show()
# print(lr.coef_)
# print (yt)
# points_x=[x/10. for x in range(-50,+50)]

# x_min, x_max = xt[:, 0].min() - .5, xt[:, 0].max() + .5
# y_min, y_max = xt[:, 1].min() - .5, xt[:, 1].max() + .5
# h = .02 
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = lr.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros((yy.ravel().shape[0],38))])
# Z = Z.reshape(xx.shape)

# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)


# line_bias = lr.intercept_
# line_w = lr.coef_.T
# points_y=[(line_w[0]*x+line_bias)/(-1*line_w[1]) for x in points_x]
# plt.plot(points_x, points_y)
# scatter = plt.scatter(xt[:,0], xt[:,1],c=yt, edgecolors='k', cmap=plt.cm.Paired)
# plt.legend()
# legend1 = plt.legend(handles=scatter.legend_elements()[0], labels = ['English','Spanish','Arabic'], loc="lower left", title="Classes")
# ax.add_artist(legend1)
# cm = (np.ones((3,3)))
# df_cm = pd.DataFrame(cm, [1,2,3],[1,2,3])
# sn.set(font_scale=1.0)
# fig, ax = plt.subplots()
# plt.figure(1, figsize=(3, 15))
# fig, ax = plt.subplots(figsize=(4,4))         # Sample figsize in inchesfig = plt.gcf()  # or by other means, like plt.subplots
# figsize = fig.get_size_inches()
# fig.set_size_inches(figsize * 5) 
# sn.heatmap(df_cm, annot=True, linewidths=.5, cbar=True)
# legend2 = plt.legend(*scatter.legend_elements(),['a','b','c'])
# plt.title("Decision boundaries and test data")
# plt.xlabel('')
# plt.ylabel('')
# cm = confusion_matrix(yt,lr.predict(xt))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(cm)

# fig, ax = plot_confusion_matrix(conf_mat=cm)
# plt.show()

# plt.tight_layout()
# plt.show()

# K Nearest Neighbours
# from sklearn.neighbors import KNeighborsClassifier
# x = x[:,:2]
# knn = KNeighborsClassifier(n_neighbors=15).fit(x,y)
# print("KNN Train:"+str(knn.score(x,y)))
# # print("KNN Test:"+str(knn.score(xt,yt)))
# print("KNN Confusion Matrix:")
# print(confusion_matrix(yt,knn.predict(xt)))
# plt.subplot()
# sn.heatmap(confusion_matrix(yt,knn.predict(xt)), annot=True,fmt='g',cmap='Greens',square=True)
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t)
# plt.show()

# x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
# y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
# # predict class using data and kNN classifier
# Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# # Plot also the training points
# ax = plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)
# classes = ['English', 'Spanish','Arabic']
# plt.legend(handles=ax.legend_elements()[0], labels=classes)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("3-Class classification (k = %i)" % (15))
# plt.show()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(x,y)
print("GNB Train:"+str(gnb.score(x,y)))
print("GNB Test:"+str(gnb.score(xt,yt)))
print("GNB Confusion Matrix:")
print(confusion_matrix(yt,gnb.predict(xt)))
# # plt.subplot()
# # sn.heatmap(confusion_matrix(yt,gnb.predict(xt)), annot=True,fmt='g',cmap='Greens',square=True)
# # b, t = plt.ylim() # discover the values for bottom and top
# # b += 0.5 # Add 0.5 to the bottom
# # t -= 0.5 # Subtract 0.5 from the top
# # plt.ylim(b, t)
# # plt.show()
ax1 = plt.scatter(x[:, 0], x[:, 1], c=y, s=50)
lim = plt.axis()
ax2 = plt.scatter(xt[:, 0], xt[:, 1], c=yt, s=20)
plt.axis(lim)
classes = ['English', 'Spanish', 'Arabic']
leg1 = plt.legend(handles=ax1.legend_elements()[0], labels=classes)
plt.title('Gaussian Naive Bayes classification')
plt.show()

# Decision Trees
# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier().fit(x,y)
# print("DTC Train:"+str(dtc.score(x,y)))
# print("DTC Test:"+str(dtc.score(xt,yt)))
# print("DTC Confusion Matrix:")
# print(confusion_matrix(yt,dtc.predict(xt)))
# plt.subplot()
# sn.heatmap(confusion_matrix(yt,dtc.predict(xt)), annot=True,fmt='g',cmap='Greens',square=True)
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t)
# plt.show()
# print(np.argsort(dtc.feature_importances_)[::-1][:5])
# dot_data = StringIO()
# export_graphviz(dtc, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('decision_tree.png')


# # Random Forests
# from sklearn.ensemble import RandomForestClassifier
#
# parameters = {'n_estimators':[3,5,7,10,13,15,20,25], 'max_depth':[None,10,20,30,40,50]}
# rfc = RandomForestClassifier()
# clf = GridSearchCV(rfc, parameters,cv=10, n_jobs=7)
# clf.fit(x,y)
# print("RandomForestClassifier Train:"+str(clf.score(x,y)))
# print("RandomForestClassifier Test:"+str(clf.score(xt,yt)))
# print("Random Confusion Matrix:")
# plt.subplot()
# sn.heatmap(confusion_matrix(yt,clf.predict(xt)), annot=True,fmt='g',cmap='Greens',square=True)
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t)
# plt.show()
# print(confusion_matrix(yt,clf.predict(xt)))
# print(clf.best_params_)
# print(clf.best_score_)
# print(np.argsort(rfc.feature_importances_)[::-1][:5])
