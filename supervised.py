# import tensorflow as tf
from keras.models import Sequential
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataset = np.load('final_dataset_top9_delta.npy')

X = dataset[:,:-1]
y = to_categorical(dataset[:,-1])
y = (dataset[:,-1])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


model = Sequential()

model.add(Dense(units=x_train.shape[1],activation='sigmoid',input_dim=x_train.shape[1]))
model.add(Dense(units=int(x_train.shape[1]/2),activation='sigmoid'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=int(x_train.shape[1]/4),activation='softmax'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=int(x_train.shape[1]/8),activation='sigmoid'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=int(x_train.shape[1]/16),activation='softmax'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=int(x_train.shape[1]/16),activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=int(x_train.shape[1]/16),activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=9,activation='softmax'))

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train, epochs=10)

evaluation = model.evaluate(x_test, y_test)

print (model.metrics_names)
print (evaluation)
