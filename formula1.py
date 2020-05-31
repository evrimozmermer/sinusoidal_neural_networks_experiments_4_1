# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:14:26 2020

@author: evrim

Test Accuracy:  0.908
"""
from sklearn import preprocessing
import numpy as np
import tqdm
from snn_conv_formula1 import SinDense
import torch as t
import pandas as pd

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

data = t.from_numpy((train_images).reshape(60000,784))
labels = t.from_numpy(train_labels_binary)

test_data = t.from_numpy((train_images).reshape(60000,784))
train_labels_binary = t.from_numpy(train_labels_binary)

sd1 = SinDense(128,last_input_condition = True,init_weights_range = [-0.1,0.1])
sd2 = SinDense(10,last_input_condition = True,init_weights_range = [-0.1,0.1])

lr = 0.001
outer = tqdm.tqdm(total=5, desc='Epoch', position=0,leave=True)
max_error_list = []
for epoch in range(5): #epoch
    accuracy = []
    pbar = tqdm.tqdm(range(60000))
    cnta = 0
    max_errors = []
    for cnt in range(60000):
        out1 = sd1.forward(data[cnt])
        out2 = sd2.forward(out1)
        
        main_error = out2-labels[cnt]
        
        error1 = sd2.backprop(main_error.unsqueeze(0),out1,learning_rate = lr)
        error2 = sd1.backprop(error1,data[cnt].unsqueeze(0),learning_rate = lr)
        
        max_error = t.max(t.abs(main_error))
        max_errors.append(max_error)
        
        if t.argmax(out2)==t.argmax(labels[cnt]):
            accuracy.append(1)
            cnta += 1
        else:
            accuracy.append(0)
            cnta += 1
        
        pbar.set_description('Epoch: {}, Sample: {}, Train Loss: {}'.format(
                    epoch,
                    cnt,
                    max_error))
        
        pbar.update(1)
    
    max_error_list.append(max_errors)
    outer.update(1)

test_accuracy = []
for cnt in range(10000):
    out1 = sd1.forward(test_data[cnt])
    out2 = sd2.forward(out1)
    
    main_error = out2-train_labels_binary[cnt]
    
    max_error = t.max(t.abs(main_error))
    max_errors.append(max_error)
    
    if t.argmax(out2)==t.argmax(train_labels_binary[cnt]):
        test_accuracy.append(1)
        cnta += 1
    else:
        test_accuracy.append(0)
        cnta += 1
        
print("\nTest Accuracy: ",np.sum(test_accuracy)/len(test_accuracy))