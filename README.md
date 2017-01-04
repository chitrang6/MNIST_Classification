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

# Descriptions about the above listed components of the Deep Learning software

1. MNIST_Dataprep.py --> This module converts the downloaded MNIST test and train data into the TFrecords which is the supported file format by the tensorflow for the dataset files. The images and labels are conveerted to this binary format and then it is provided as input. This will act as one type of encoding the dataset.

2. inputToMNISTModel.py --> This module provides the input to our Deep Learning model. Here LeNEt is selected as the Deep Learning model for the hand-written image classification. It decodes the TfRecords files both for the testing and training data-set and then it is fed to the LeNet model. In this way we can handle a very large dataset.

3. model_definition.py --> As described above this is very nice moduler design for the software. This file contains the LeNet model definition. Anyone can just change the model parameters directly in this file and they don't have to change the model parameters seprately for training adnd validation.

4. Train_MNIST_Model.py --> This module contains the code for the training of the LeNet model on the MNIST dataset. It uses all the above moduler implementation for the training. It will aslo save the summary of the training process and one can visualize the training process using the tensorboard.

5. Validation_MNIST.py --> Classification accuracy with the Streaming MSE is used here as to measure the model performance. This file contains the implementation for the validating the trainied LeNEt model using the Testing data-set. 

6. predictions_usingModel.py --> This module contains the implementation to predict the digits or to classify the provided input image as one of the 0-9 digits for the unlown images which are not known to Model. 

7. predictions.py --> This file is used for making the prediciotns on the unlown input image. The input is provided as the URL form as the command line argument. This software supports for now only .png or .jpeg images. How to use this Deep Learning software is defined below.


# Version 1.0

## $ python predictions.py <"image url">


## $ python predictions.py  "http://www.fastforwardlabs.com/blog-images/miriam/mnist_%5B6%5D.png"

The output of the program displays the probability of having the perticular digit in the input image. The sample output is given below for the above image url.


The URL is: http://www.fastforwardlabs.com/blog-images/miriam/mnist_%5B6%5D.png

Image is a png

coming here

[[  2.52584718e-19   1.06800495e-22   8.18386664e-16   1.51070594e-33
    3.93154816e-16   1.06397777e-17   1.00000000e+00   5.68210753e-33
    2.67818678e-26   4.87909434e-25]]

6

The probabilities are for the digits 0-9 respectively.
Where 6 is the digit classified fo the given input image by the trained model.


In this software repo, the usage_example.png file contains the screenshot of the abve command executed on the MacOS X and with the result.

------------------------------------------------------------------------------------------------------------------------------

# Version 2.0



Thank you
