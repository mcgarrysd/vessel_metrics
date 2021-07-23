#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:39:34 2020

Tensorflow tutorial
https://www.youtube.com/watch?v=6g4O5UOH304

@author: sean
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

plt.close("all")

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# print(train_data[0])

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0 
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# print(decode_review(test_data[0]))

print(len(test_data[0]), len(test_data[1]))

model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = "relu"))
model.add(keras.layers.Dense(1,activation = "sigmoid"))

