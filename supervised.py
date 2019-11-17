# import tensorflow as tf
from keras.models import Sequential
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

<<<<<<< HEAD
dataset = np.load('final_dataset.npy')
# dataset = np.load('final_dataset.npy')
# dataset = dataset[dataset[:,-1] != 2]
# dataset = dataset[dataset[:,-1] != 9]
# dataset = dataset[dataset[:,-1] != 8]
# dataset = dataset[dataset[:,-1] != 7]
=======
dataset = np.load('final_dataset_top9_delta.npy')
>>>>>>> 3604996304932aabd1e299cd4e82925dd3d24a50

X = dataset[:,:-1]
y = to_categorical(dataset[:,-1])
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


model = Sequential()

model.add(Dense(units=64,activation='relu',input_dim=x_train.shape[1]))
# model.add(Dense(units=int(x_train.shape[1]/2),activation='sigmoid'))
# model.add(Dropout(rate=0.3))
# model.add(Dense(units=int(x_train.shape[1]/4),activation='softmax'))
# model.add(Dropout(rate=0.3))
# model.add(Dense(units=int(x_train.shape[1]/8),activation='sigmoid'))
# model.add(Dropout(rate=0.3))
# model.add(Dense(units=48,activation='softmax'))
# model.add(Dropout(rate=0.1))
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(units=16,activation='relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(units=y.shape[1],activation='softmax'))

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train, epochs=10)

evaluation = model.evaluate(x_test, y_test)

print (model.metrics_names)
print (evaluation)
<<<<<<< HEAD

# print (y_test,model.predict(x_test))
# print (np.argmax(y_test),model.predict_classes(x_test))
print (confusion_matrix(np.argmax(y_test, axis = 1), model.predict_classes(x_test)))

=======
>>>>>>> 3604996304932aabd1e299cd4e82925dd3d24a50
