#!/usr/bin/python

import sys
from predictions_usingModel import *
print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)




url = str(sys.argv[1])
print "The URL is: " + url
MNIST_predictions(url)




