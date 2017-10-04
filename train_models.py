import numpy as np

import keras
from keras.models import load_model
from tensorflow.python.platform import app
from keras.utils import np_utils

from Models import cnn, sota
import helpers
import os

import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_string('mode', 'train', '(train,test,finetune)')
flags.DEFINE_string('dataset', 'cifar100', '(cifar100,svhn,mnist)')
flags.DEFINE_float('learning_rate', 0.01 ,'Learning rate for classifier')
flags.DEFINE_string('save_here', 'saved_model', 'Path where model is to be saved')


def train_logit_proxy(X_train, Y_train, nb_classes, learning_rate, shape, num_epochs=100, train_temp=1):
	# As defined here: https://github.com/dribnet/kerosene/blob/master/examples/cifar100.py
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode='same',
			input_shape=(shape[0], shape[1], shape[2])))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))
	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(nb_classes))

	def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

	sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

	model.fit(data.train_data, data.train_labels,
              batch_size=16,
              validation_split=0.2,
              epochs=num_epochs,)

	return model


def main(argv=None):
	"""
    Train a network using defensive distillation.

    Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
    Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
    IEEE S&P, 2016.
    """
	n_classes = 10
	shape = (3, 32, 32)
	tf.set_random_seed(1234)

	if FLAGS.dataset == 'cifar100':
		n_classes = 100
	elif FLAGS.dataset == 'mnist':
		shape = (1, 28, 28)
	elif FLAGS.dataset == 'svhn':
		pass
	else:
		print "Invalid dataset specified. Exiting"
		exit()

	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	#Don't hog GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	# evaluate the labels (normal model with softmax; temperature=1)
    predicted = teacher.predict(X_train)
    # Y_train = sess.run(tf.nn.softmax(predicted/train_temp))
    Y_train = predicted

    # train the student model at temperature t
    student = train_logit_proxy(X_train, Y_train, n_classes, FLAGS.learning_rate. shape, FLAGS.nb_epochs, train_data)

    # and finally we predict at temperature 1
    predicted = student.predict(data.train_data)

    print(predicted)

    # save student model
    student.save(FLAGS.save_here)

if __name__ == '__main__':
	app.run()
