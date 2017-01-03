# Author:- Chitrang Talaviya
# This file for Training the LeNet model for the MNIST digits classification.



import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from model_definition import *
from inputToMNISTModel import *
import shutil


cwd = os.getcwd()
train_dir = cwd +  '/log/train/'

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)

def main():
	tf.logging.set_verbosity(tf.logging.INFO)
	images , labels = InputsToMNISTModel('TRAIN.tfrecords' , 32 , None , one_hot_labels = True)
	predictions , _ = LeNet_Model(images)
	slim.losses.softmax_cross_entropy(predictions, labels)
	total_loss = slim.losses.get_total_loss()
	tf.scalar_summary('loss', total_loss)
	optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
	train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)
	slim.learning.train(train_op, train_dir ,number_of_steps=20000,  save_summaries_secs=1 ,save_interval_secs=1)


if __name__ == '__main__':
	main()