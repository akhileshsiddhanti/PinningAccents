import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import re
from keras.utils import to_categorical
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout

warnings.simplefilter(action='ignore', category=FutureWarning)

ENG_SIZE = 150

images = os.listdir('Images')
data = []
labels = []
unique_accents = np.load('unique_accents_top3_delta.npy')

# Loading in the data and corresponding labels
for img in images:
	accent = re.split('\d',img)[0]
	image = cv2.imread('Images/'+img)
	image = np.expand_dims(cv2.resize(image,(128,128)),0)
	if len(data)==0:
		data = image
		labels = accent
	else:
		data = np.vstack((data,image))
		labels = np.hstack((labels,accent))

data = data.astype('float32')/255
labels = np.array([np.where(unique_accents==x)[0][0] for x in labels])

few_eng = np.random.permutation(np.argwhere(labels==1))[:ENG_SIZE][:,0]
no_eng = labels!=1

eng_data, eng_y = data[few_eng], labels[few_eng]
no_eng_data, no_eng_y = data[no_eng], labels[no_eng]


crop_data = np.vstack((eng_data, no_eng_data))
crop_y = np.hstack((eng_y, no_eng_y))

x, xt, y, yt = train_test_split(crop_data,crop_y,test_size=0.2, shuffle=True, stratify=crop_y, random_state=42)

eng = (yt==1)
spanish = (yt==2)
arabic = (yt==0)

y = to_categorical(y)
yt = to_categorical(yt)	



# Model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=data.shape[1:]))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))

model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

print(model.summary())

model.fit(x,y,epochs=10,batch_size=8,validation_split=0.15,shuffle=True)
print(model.evaluate(x,y))
print(model.evaluate(xt,yt))
print(model.evaluate(xt[eng],yt[eng]))
print(model.evaluate(xt[spanish],yt[spanish]))
print(model.evaluate(xt[arabic],yt[arabic]))
print(yt)
print(np.argmax(model.predict(xt),1))
