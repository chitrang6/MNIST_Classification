# The first task is to read the mnist dataset.
#
#

import os
import struct
import numpy as np
from pylab import *
import tensorflow as tf



_NUMBER_CHANNESL = 1

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
	label_unpack.read(4)
	total = label_unpack.read(4)
	total = struct.unpack('>I', total)[0]
	#magic_num , total = unpack(">II", label_unpack.read(8))
	#labels_asNumpy = np.readfile(label_unpack , dtype = np.int8) # use the numpy.readfile as we are reading from file.

	img_unpack = open(image_file , 'rb')
	img_unpack.read(4)
	total = img_unpack.read(4)
	total = struct.unpack('>I', total)[0]
	rows = img_unpack.read(4)
	rows = struct.unpack('>I', rows)[0]
	cols = img_unpack.read(4)
	cols = struct.unpack('>I', cols)[0]
	#magic_num , total , rows , cols = unpack(">IIII" , img_unpack.read(16))
	#img_asNumpy = np.fromfile(img_unpack , dtype = np.uint8).reshape(len(labels_asNumpy) , rows , cols) # use the numpy.readfile as we are reading from file.


	x = np.zeros((total, rows, cols , 1), dtype=np.float32)  # Initialize numpy array
	y = np.zeros((total, 1), dtype= np.uint8)  # Initialize numpy array

	for i in range(total):
		for row in range(rows):
			for col in range(cols):
				tmp_pixel = img_unpack.read(1)  # Just a single byte
				#print tmp_pixel
				tmp_pixel = struct.unpack('>B' , tmp_pixel)
				x[i][row][col] = tmp_pixel[0]
		tmp_label = label_unpack.read(1)
		tmp_label = struct.unpack('>B' , tmp_label)
		y[i] =tmp_label[0]
	close(label_file) # Closing both the image and label files.
	close(image_file)
	return (x, y)

	#each_image = lambda item: (labels_asNumpy[item] , img_asNumpy[item])
	#print labels_asNumpy.size
	# Create an iterator which returns each image in turn
	#for i in xrange(len(labels_asNumpy)):
	#	yield each_image(i)


def ConvertToOneHot(labels , num_classes = 10):
	Total_labels = labels.shape[0]
	index = np.arange(Total_labels) * num_classes
	label_with_one_hot = np.zeros((Total_labels , num_classes), dtype=np.uint8)
	label_with_one_hot.flat[index + labels.ravel()] = 1
	return label_with_one_hot




train_image , train_label  = readMNISTDataset("train")
#test_image , test_label = readMNISTDataset("test")
#one_hot_coded = ConvertToOneHot(test_label)
print train_image[0]

def bytes_feature(values):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def int64_feature_Numpy(values):
	#if not isinstance(values, (tuple, list)):
	#	values = [values]
	#print values
	if isinstance(values, np.ndarray):
		values = values.tolist()
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))

def int64_feature(values):
	#if not isinstance(values, (tuple, list)):
	#	values = [values]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))

def ConvertToTFRecords(image , label , name):
	""" 
	This function is kind of helper function to convert the extracted data-set to the 
	Tensorflow TFRecords format.

	"""

	numOfExample = image.shape[0]
	shape = (image.shape[1] , image.shape[2] , 1) # As this are the gray scale images
	filename = os.path.join(".", name + '.tfrecords')
	with tf.python_io.TFRecordWriter(filename) as tfWriter:
		with tf.Graph().as_default():
			Single_image_holder = tf.placeholder(dtype = tf.uint8 , shape = shape)
			encoded_png = tf.image.encode_png(Single_image_holder)
			with tf.Session('') as sess:
				for j in range(numOfExample):
					sys.stdout.write('\r>> Converting image %d/%d' % (j + 1 , numOfExample))
					sys.stdout.flush()

					png_str = sess.run(encoded_png , feed_dict = {Single_image_holder:image[j]})

					#example = dataset_utils.image_to_tfexample(png_str , 'png' , image.shape[1] , image.shape[2] , label[j])
					examples = tf.train.Example(features = tf.train.Features(
						feature = {'image/encode': bytes_feature(png_str),
									'image/format': bytes_feature('png'),
									'image/class/label' : int64_feature(int(label[j])),
									'image/height' : int64_feature(image.shape[1]),
									'image/width' : int64_feature(image.shape[2])}))
					tfWriter.write(examples.SerializeToString())
	tfWriter.close()



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

#print test_image.shape
#ConvertToTFRecords(train_image , train_label , "TRAIN")

#print train_label.shape
#print one_hot_coded.shape
#print one_hot_coded[0]


