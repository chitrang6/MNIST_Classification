# Author:- Chitrang Talaviya
# This file reads the TFrecords file and provides the input to the MNIST LeNet model.
# This will act as queue for the MNIST model. It can read and decode both the Training and test data so that these 
# functions are going to be used by both training and the validation.


import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

def Read_ExampleandDecode(TFRecord_name):
	TFreader = tf.TFRecordReader()
	_, serial_example = TFreader.read(TFRecord_name)
	features = tf.parse_single_example(
		serial_example , 

		features = {
		'image_raw':  tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64)
		#'height': tf.FixedLenFeature([], tf.int64),
		#'width': tf.FixedLenFeature([], tf.int64),
		#'depth': tf.FixedLenFeature([], tf.int64)
		})

	#image = tf.image.decode_png(features['image/encode'] , dtype = tf.uint8)
	#print image
	# Convert from [0, 255] -> [-0.5, 0.5] floats.
	#image = tf.image.decode_image(features['image/encode'])
	#label = tf.cast(features['image/class/label'], tf.int32)
	#height = tf.cast(features['height'], tf.int32)
	#width = tf.cast(features['width'], tf.int32)
	#epth = tf.cast(features['depth'], tf.int32)
	#print image
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	label = tf.cast(features['label'], tf.int32)
	image.set_shape([784])
	image = tf.cast(image , tf.float32) * (1.0 /255) - 0.5
	image = tf.reshape(image, [28, 28, 1])
	#image.set_shape([784])
	#image = tf.cast(image , tf.float32) * (1.0 /255) - 0.5
	#image = tf.reshape(image, [28, 28, 1])
	#print image
	#print label
	return image, label

# This function will prepare inputs for the Deep Learning model in the tensorflow.

NUM_CLASSES = 10


def InputsToMNISTModel(record_name , batch_size , num_epochs , one_hot_labels = False):
	""" This function will read the input data from num_epochs times."""
	if not num_epochs:
		num_epochs = None

	filepath = os.path.join("." , record_name)
	print filepath

	with tf.name_scope('input'):
		QueueForDatasetPath = tf.train.string_input_producer([filepath] , num_epochs = num_epochs)



		image , label = Read_ExampleandDecode(QueueForDatasetPath)

		if one_hot_labels:
			label = tf.one_hot(label, NUM_CLASSES, dtype=tf.int32)



		# Now, the task is to define the batch size and shuffle the dataset according the batch size givent by the user.\
		# This task will run using the two threads.

		images , labels = tf.train.batch([image, label] , batch_size = batch_size ,
		                num_threads = 2 , capacity = 3*batch_size)




	return images , labels
