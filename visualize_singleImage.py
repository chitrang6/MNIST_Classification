import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import tensorflow.contrib.slim as slim
from scipy import misc
import urllib2
import os


batch_size = 3
image_size = 28


def preprocess_image(image, output_height, output_width, is_training):
        #image = tf.to_float(image)
        #image = tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)
        image = tf.image.resize_images(image, (output_width, output_height), method = 0)  
        #image = tf.div(image, 255)
        #image = tf.sub(image, 0.5)
        return image

with tf.Graph().as_default():
        url = 'https://i.stack.imgur.com/c4JXq.png'
        image_string = urllib2.urlopen(url).read()
        image = tf.image.decode_png(image_string, channels=1)     # This will decode RGB image.
        process_image  = preprocess_image(image , image_size , image_size  , is_training = False)
        with tf.Session() as sess:
                tf.initialize_all_variables().run()
                np_image = sess.run([process_image])
                print np_image
        plt.figure()
        np_image = np.reshape(np_image , 784)
        np_image = np.reshape(np_image , (28, 28))
        plt.imshow(np_image.astype(np.uint8))
        plt.axis('off')
        plt.show()