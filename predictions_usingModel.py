#
#
# This file contains code for making the prediction using the trained model. The user gives the URL of the image file and 
# the implementation of this code will read that url containing the image and makes the prediction of the digit. The predictions are 
# in the form of the probability of having that digit in the image. 
#

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import tensorflow.contrib.slim as slim
from model_definition import *
from scipy import misc
import urllib2 ,cStringIO
import os
from PIL import Image , ImageOps
import re

IMAGE_SIZE_HEGIHT = 28
IMAGE_SIZE_WIDTH = 28

cwd = os.getcwd()
model_file_path = cwd +  '/log/train/model.ckpt-20000'




# The names of the classes.
_CLASS_NAMES = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
]






print cwd
print model_file_path

def preprocess_image_forpredictions(image, output_height, output_width, is_training , isBackgroundBlack = True):
        image = tf.to_float(image)
        #image = tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)
        image = tf.image.resize_images(image, (output_width, output_height), method = 0)
        if isBackgroundBlack: 
            image = tf.div(image, 255)
            image = tf.sub(image, 0.5)
        else:
            print "coming here"
            image = tf.mul(image , -1)
            image = tf.add(image, 255)
            image = tf.div(image, 255)
            image = tf.sub(image, 0.5)
        return image






def MNIST_predictions(url):

#url = "http://www.fastforwardlabs.com/blog-images/miriam/mnist_%5B6%5D.png"

    mnist_file = cStringIO.StringIO(urllib2.urlopen(url).read())
    #im = Image.open(mnist_file).convert('LA')
    im = Image.open(mnist_file)
    im = ImageOps.grayscale(im)
    #pixels = im.getdata()

    black_thresh = 30

    black = 0
    n = 0
    isBackgroundBlack = False
    for pixel in im.getdata():
        n = n+ 1
        if pixel == 0:
            black+=1
    if (black / float(n)) > 0.4:
        isBackgroundBlack = True
        print("mostly black")
        #inverted_image = PIL.ImageOps.invert(im)
    else:
        inverted_image = ImageOps.invert(im)
        inverted_image.save('inverted_image.png')


    #img = Image.open('result.png')
    #image_string = img.tobytes()

    isImageJPEG = False

    request = urllib2.urlopen(url)
    mime = request.info()['Content-type']

    if mime.endswith("png"):
        print("Image is a png")
    elif mime.endswith("jpeg"):
        print("Image is a jpeg")
        isImageJPEG = True
    else:
        raise ValueError('Please give URL of the png or jpeg images. Currently version of the software supports only png or jpeg images.')



    with tf.Graph().as_default():
            #url = 'http://neuralnetworksanddeeplearning.com/images/mnist_complete_zero.png'
            image_string = urllib2.urlopen(url).read()
            #mnist_file = cStringIO.StringIO(urllib2.urlopen(url).read())
            if not isImageJPEG:
                image = tf.image.decode_png(image_string, channels=1)     # This will decode RGB image.
                image2 = tf.image.decode_png(image_string, channels=1)     # This will decode RGB image.
            else:
                image = tf.image.decode_jpeg(image_string, channels=1)     # This will decode RGB image.
                image2 = tf.image.decode_jpeg(image_string, channels=1)     # This will decode RGB image.



            process_image  = preprocess_image_forpredictions(image , IMAGE_SIZE_HEGIHT , IMAGE_SIZE_WIDTH  , is_training = False , isBackgroundBlack = isBackgroundBlack)
            process_image  = tf.expand_dims(process_image, 0)
            image2 = tf.image.resize_images(image2, (IMAGE_SIZE_HEGIHT, IMAGE_SIZE_WIDTH), method = 0) 
            with slim.arg_scope(lenet_arg_scope()):
                    logits, _ = LeNet_Model(process_image , is_training=False)
            probabilities = tf.nn.softmax(logits)
            init_op = tf.initialize_all_variables()
            init_fn = slim.assign_from_checkpoint_fn(model_file_path,
                    slim.get_model_variables('LeNet'))
            with tf.Session() as sess:
                    #tf.initialize_all_variables().run()
                    init_fn(sess)
                    np_image, probabilities = sess.run([image2, probabilities])
                    #print np_image.shape
                    #print probabilities
                    probabilities_sort = probabilities[0, 0:]
                    probabilities_list = probabilities.tolist()
                    #print probabilities.shape
                    #print probabilities
                    #probabilities_dict = dict(zip(_CLASS_NAMES , probabilities.T))
                    #probabilities_dict = {}
                    #for i in range(len(_CLASS_NAMES)):
                    #    print _CLASS_NAMES[i]
                    #    probabilities_dict[_CLASS_NAMES[i]] = probabilities[0][i]
                    #print probabilities_dict
                    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities_sort), key=lambda x:x[1])]
                    #print type(sorted_inds[0])
                    print sorted_inds[0]
            #plt.figure()
            #np_image = np.reshape(np_image , 784)
            #np_image = np.reshape(np_image , (28, 28))
            #plt.imshow(np_image.astype(np.uint8))
            #plt.axis('off')
            #plt.show()
    return sorted_inds[0] , probabilities_list