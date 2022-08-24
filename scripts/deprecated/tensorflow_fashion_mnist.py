#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 08:53:50 2020

Tensorflow tutorial
https://www.youtube.com/watch?v=6g4O5UOH304

@author: sean
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.imshow(train_images[7])
plt.show()

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
        keras.layers.Flatten(input_shape = (28,28)),
        keras.layers.Dense(128, activation = "relu"),
        keras.layers.Dense(10, activation="softmax")
        ])

model.compile(optimizer = "adam", loss ="sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs = 5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel("actual: " + class_names[test_labels[i]])
    plt.title("Prediction " + class_names[np.argmax(prediction[0])])
    plt.show()
    
print(class_names[np.argmax(prediction[0])])
