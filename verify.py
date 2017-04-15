## verify.py -- check the accuracy of a neural network
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

import tensorflow as tf
import numpy as np

BATCH_SIZE = 1

with tf.Session() as sess:
    data, model = MNIST(), MNISTModel("models/mnist", sess)
    data, model = CIFAR(), CIFARModel("models/cifar", sess)
    data, model = ImageNet(), InceptionModel(sess)

    x = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
    y = model.predict(x)

    r = []
    for i in range(0,len(data.test_data),BATCH_SIZE):
        pred = sess.run(y, {x: data.test_data[i:i+BATCH_SIZE]})
        #print(pred)
        #print('real',data.test_labels[i],'pred',np.argmax(pred))
        r.append(np.argmax(pred,1) == np.argmax(data.test_labels[i:i+BATCH_SIZE],1))
        print(np.mean(r))
