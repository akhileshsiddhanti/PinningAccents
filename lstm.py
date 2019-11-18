# import tensorflow as tf
from keras.models import Sequential
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import plot_model
import matplotlib.pyplot as plt

dataset = np.load('final_dataset_top3_delta.npy')
# dataset = np.load('final_dataset.npy')
# dataset = dataset[dataset[:,-1] != 2]
# dataset = dataset[dataset[:,-1] != 9]
# dataset = dataset[dataset[:,-1] != 8]
# dataset = dataset[dataset[:,-1] != 7]


X = dataset[:,:-1]


y = to_categorical(dataset[:,-1])
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, stratify=y, random_state=0)
data_dim = x_train.shape[1]


x_train = x_train[:,np.newaxis]
x_test = x_test[:,np.newaxis]
print (x_train.shape, y_train.shape)


timesteps = 1
model = Sequential()

# model.add(Embedding(1000, 64, input_length=x_train.shape[1], dropout=0.1))
# model.add(Dense(units=64, input_dim=x_train.shape[1],activation='relu'))
# model.add(Dense(units=20,activation='relu'))
# model.add(Dropout(units=20))
model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim), dropout = 0.1) )
model.add(LSTM(20))
# model.add(Dense(units=20,activation='relu'))
model.add(Dense(units=y.shape[1],activation='softmax'))

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train, epochs=100, validation_split=0.2)
# history = model.fit(x_train,y_train, epochs=100)
evaluation = model.evaluate(x_test, y_test)

# plot_model(model, to_file='lstm.png')

print (model.metrics_names)
print (evaluation)

# print (y_test,model.predict(x_test))
# print (np.argmax(y_test),model.predict_classes(x_test))
print (confusion_matrix(np.argmax(y_test, axis = 1), model.predict_classes(x_test)))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xticks(np.arange(0, 100, 5))
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()