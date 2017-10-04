## setup_cifar.py -- cifar data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import tensorflow as tf
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

from keras.datasets import cifar10
from keras.utils import np_utils

class CIFAR:
	def __init__(self):
	img_rows = 32
	img_cols = 32
	nb_classes = 10
	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	self.train_data = X_train
	self.train_labels = Y_train
	self.test_data = X_test
	self.test_labels = Y_test


class CIFARModel:
	def __init__(self, restore, session=None):
		self.num_channels = 3
		self.image_size = 32
		self.num_labels = 10

		model = Sequential()

		model.add(Conv2D(64, (3, 3),
								input_shape=(32, 32, 3)))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3)))
		model.add(Activation('relu'))
		model.add(Conv2D(128, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dense(10))

		model.load_weights(restore)

		self.model = model

	def predict(self, data):
		return self.model(data)
