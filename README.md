# MNIST_Classification

This is the git repository for the MNIST hand-written digits classification using Deep Learning framework Tensorflow. This is basically a multi-class classification probelm. The user will provide the url of the image containg digits and this dee p learning software will classify the image and returns the result in the probabilities. The output probabilities shows the probability of having the perticular digits from 0-9 in the given input image. Again the image is provided as the url as the command line argument. How to use this software is defined later. 

This software contains the implementation to download the MNIST dataset, convert it to TFrecords which is the supported file format to encode the images and lables. It also contains the implementation for the training of the LeNet deep learning model for the MNIST hand written digits classification. Validating this trainied model is also there in this software. This software uses the Tensorflow-slim library for the training validation and make predictions.

## How to use this software?

* This softwar is written in python and contains modular design. The following are the main components of this software.

1) MNIST_Dataprep.py

2) inputToMNISTModel.py

3) model_definition.py

4) Train_MNIST_model.py

5) Validation_MNIST.py

6) predictions_usingModel.py

7) predictions.py 




