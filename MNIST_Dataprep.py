# Author:- Chitrang Talaviya
# This file Download the mnist dataset and convert it the Numpy array. After converting to  the numpy array this file 
# will convert the numpy array into the TFrecords for the test and train dataset.
# Output:- Two TFrecords file i) TRAIN.tfrecords  ii) TEST.tfrecords

import os
import struct
import numpy as np
from pylab import *
import tensorflow as tf

def read32(input):
	dt = np.dtype(np.uint32).newbyteorder('>')
	return np.frombuffer(input.read(4), dtype=dt)[0]

_NUMBER_CHANNESL = 1
_MAGIC_NUM_FORMNIST_LABEL = 2049
_MAGIC_NUM_FORMNIST_IMAGE = 2051


def bytes_feature(values):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def int64_feature(values):
	#if not isinstance(values, (tuple, list)):
	#	values = [values]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))



def readMNISTDataset(dataset = "train" , path = "."):
	""" This function will read the training and testing dataset downloaded from ...
		website. It will read those file formats and store each image and label in the dataset as numpy 
		array. As mentioned in the website the 
	"""
	if dataset is "train":
		image_file = os.path.join(path, 'train-images-idx3-ubyte')
		label_file = os.path.join(path, 'train-labels-idx1-ubyte')
	elif dataset is "test":
		image_file = os.path.join(path, 't10k-images-idx3-ubyte')
		label_file = os.path.join(path, 't10k-labels-idx1-ubyte')
	else:
	    raise ValueError, "Invalid Entry try -> training and testing keywords."

	# Now, the task is to unpack this packed structure format to numpy array
	# As written in the data-set web site: All the integers in the files are stored in the MSB first (high endian) format 
	# used by most non-Intel processors.
	# Users of Intel processors and other low-endian machines must flip the bytes of the header.
	# We need to flip the bytes.

	# First we will read the label. will filp the bytes using the ">I"
	# Read the labels Here read the magic number and number of labels first.

	label_unpack = open(label_file , 'rb')
	#label_unpack.read(4)
	magic_num = read32(label_unpack)

	if magic_num != _MAGIC_NUM_FORMNIST_LABEL:
		raise ValueError('Invalid Magic Number for MNIST dataset labels.')

	#total = label_unpack.read(4)
	#total = struct.unpack('>I', total)[0]
	num_items = read32(label_unpack)
	buff = label_unpack.read(num_items)
	labels = np.frombuffer(buff , dtype = np.uint8)


	#magic_num , total = unpack(">II", label_unpack.read(8))
	#labels_asNumpy = np.readfile(label_unpack , dtype = np.int8) # use the numpy.readfile as we are reading from file.

	#img_unpack = open(image_file , 'rb')
	#img_unpack.read(4)
	#total = img_unpack.read(4)
	#total = struct.unpack('>I', total)[0]
	#rows = img_unpack.read(4)
	#rows = struct.unpack('>I', rows)[0]
	#cols = img_unpack.read(4)
	#cols = struct.unpack('>I', cols)[0]
	#magic_num , total , rows , cols = unpack(">IIII" , img_unpack.read(16))
	#img_asNumpy = np.fromfile(img_unpack , dtype = np.uint8).reshape(len(labels_asNumpy) , rows , cols) # use the numpy.readfile as we are reading from file.
	img_unpack = open(image_file , 'rb')
	magic_num = read32(img_unpack)
	if magic_num != _MAGIC_NUM_FORMNIST_IMAGE:
		raise ValueError('Invalid Magic Number for MNIST dataset images.')

	num_images = read32(img_unpack)
	rows = read32(img_unpack)
	cols = read32(img_unpack)
	buf = img_unpack.read(rows * cols * num_images)
	images = np.frombuffer(buf, dtype=np.uint8)
	images = images.reshape(num_images, rows, cols, 1)

	return (images, labels)


# The following execution will Conver the MNIST downloaded datase file to numpy array for both image and lables.

test_image , test_label  = readMNISTDataset("test")
train_image , train_label  = readMNISTDataset("train")
#test_image , test_label = readMNISTDataset("test")
#one_hot_coded = ConvertToOneHot(test_label)






def ConvertToTFRecords_Modified(image , label , name):
	""" 
	This function is kind of helper function to convert the extracted data-set to the 
	Tensorflow TFRecords format.

	"""

	numOfExample = image.shape[0]
	shape = (image.shape[1] , image.shape[2] , 1) # As this are the gray scale images
	#print shape
	filename = os.path.join(".", name + '.tfrecords')
	with tf.python_io.TFRecordWriter(filename) as tfWriter:
		for index_through_single_image in range(numOfExample):
			#print image[index_through_single_image].shape
			image_raw = image[index_through_single_image].tostring()
			#print image_raw.shape
			examples = tf.train.Example(features = tf.train.Features(
				feature = {'height': int64_feature(image.shape[1]),
							'width': int64_feature(image.shape[2]),
							'depth' : int64_feature(1),
							'label' : int64_feature(int(label[index_through_single_image])),
							'image_raw' : bytes_feature(image_raw)}))
			tfWriter.write(examples.SerializeToString())
	tfWriter.close()


# The following lines will convert the numpy image stack to the TFrecords which is the supported file format for the Tensorflow.


ConvertToTFRecords_Modified(train_image , train_label , "TRAIN")
ConvertToTFRecords_Modified(test_image , test_label , "TEST")