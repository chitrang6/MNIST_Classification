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

This version now contains the full implementation of the MNIST classification using the Flask Web Server. The User will run the .html file from the local machine and also run the web server locally created with the Python Flask. Then the user will provide the image URL, he/she wants to classify the digit and the web server will run the Deep Learning classification task and return the digits identified by the Deep Learning algorithm (Classifier). It will show to the web-client the correct classification and also return the probabilities as the JSON blob.

## How to run the Version 2.0?

To run the Version 2.0 of the software, one should first start the web server. The web -server contains the Deep Learinng algorithm to classify the hand-written digits. The web-server calls the prediction API talked in the Version 1.0 of the software.

-- How to run the Web-server (Flask)?

### $ python MNIST_ClassifierServer.py

Execute the above command in the command line. It will start the web-server written in the Python Flask. 


-- Now, the next step is to start the Web-client to send the input image request to the Server and get the results from the Web-server as POST request.

Steps to run the Web-Client

### 1) Open your browser.

### 2) Type 127.0.0.1:5000. You will see the home page asking to input the Image URL. Click on that underlined line.

### 3) Now, the next web-page will open. In that next page, user will be asked to input the image URL. 

### 4) Give the image URL in the 'URL' field and click the Classify Button.

### 5) After clicking the Classify button, The client sent the request to the web-server, the web server will get the image URl and will run the Deep LEarning algorithm to Classify the image.

### 6) The output as a response from the Web-server is displayed as the digit classified by the web-server. It will aslo return the probabilities and the results in the form of JSON blob.


## Additional Implementation:

--> This Deep Learning algorithm (classsifier) for the MNIST hand written digit classification can handle both the images of having balck and also images having white background.



Thank you



Thank you
