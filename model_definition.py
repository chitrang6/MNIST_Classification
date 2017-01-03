# This file contains the moedl definition used for the MNIST image classification
# if you want to change the model parameters just chnage and play around with your model.
# Here, I have used the popular LeNet Model for the MNIST classification.


import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import tensorflow.contrib.slim as slim



def lenet_arg_scope(weight_decay=0.0):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
      activation_fn=tf.nn.relu) as sc:
                return sc

def LeNet_Model(images, num_classes = 10,  is_training=False, dropout_keep_prob=0.5, prediction_fn=slim.softmax,scope='LeNet'):
        end_points = {}
        with tf.variable_scope(scope , 'LeNet', [images , num_classes]):

                net = slim.layers.conv2d(images, 32, [5,5], scope='conv1')
                net = slim.layers.max_pool2d(net, [2,2] , 2, scope='pool1')
                net = slim.layers.conv2d(net, 64, [5,5], scope='conv2')
                net = slim.layers.max_pool2d(net, [2,2], 2 , scope='pool2')
                net = slim.layers.flatten(net)
                end_points['Flatten'] = net
                net = slim.fully_connected(net, 1024, scope='fc3')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout3')
                logits = slim.fully_connected(net, num_classes, activation_fn=None,scope='fc4')
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
        return logits, end_points