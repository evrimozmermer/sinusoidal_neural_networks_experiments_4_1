import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
import pandas as pd
from sklearn import preprocessing

df_train = pd.read_csv('../../digit_dataset/mnist_train.csv').values
train_images = df_train[:,1:]/255+0.01
train_images = np.array(train_images,dtype=np.float32)
train_labels = df_train[:,0].reshape((train_images.shape[0],1))
lb = preprocessing.LabelBinarizer()
lb.fit(train_labels)
train_labels_binary = lb.transform(train_labels)
train_labels_binary = np.asarray(train_labels_binary)

df_test = pd.read_csv('../../digit_dataset/mnist_test.csv').values
test_images = df_test[:,1:]/255+0.01
test_labels = df_test[:,0]
test_labels_binary = lb.transform(test_labels)

model = Sequential([
        Dense(128, activation='relu'),
        Dense(10, activation='sigmoid'),
])

model.compile(SGD(lr=.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
  train_images,
  train_labels_binary,
  batch_size=1,
  epochs=1,
  validation_data=(test_images, test_labels_binary),
)