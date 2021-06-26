from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import collections

name_t=collections.namedtuple('ex',['x','y'])
example=name_t._make([10,20])
print(example.x,example.y)
example=example._replace(x=100)
print(example.x,example.y)


import tensorflow.contrib.rnn.python.ops.rnn_cell




# tf.contrib.layers.batch_norm
x=tf.constant([[[1,2,3],[4,5,6]],
                  [[7,8,9],[10,11,12]]])
y=tf.reduce_mean(x)
y_0=tf.reduce_mean(x,axis=0)
y_1=tf.reduce_mean(x,axis=1)
y_2=tf.reduce_mean(x,axis=2)

with tf.Session() as sess:
    print(y.eval())
    print(y_0.eval())
    print(y_1.eval())
    print(y_2.eval())


