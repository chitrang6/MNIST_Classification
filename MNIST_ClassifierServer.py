from flask import Flask, render_template, request, jsonify
import requests
import httplib2
import json
from predictions_usingModel import *
import mimetypes, urllib2



app = Flask(__name__)


class MNIST_classification:
	def __init__(self, imageURL , ClassificationResultDigit , probability_list):
		self.imageURL = imageURL
		self.ClassificationResultDigit = ClassificationResultDigit
		self.probability_list = probability_list


	def toJSON(self):
		return {"MNIST_image" : {'imageURL' : self.imageURL, 'ClassificationResultDigit' : self.ClassificationResultDigit ,
		'probability_list': self.probability_list}}


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/classifier')
def classifier():
	return render_template('classifier.html')

@app.route('/process', methods=['POST'])
def process():
	valid_url = False
	image_url = False
	inputUrl = request.form['name']
	headers={
            "Range": "bytes=0-10",
            "User-Agent": "MyTestAgent",
            "Accept":"*/*"
	}
	
	req = urllib2.Request(inputUrl, headers=headers)
	response = urllib2.urlopen(req)
	if response.code in range(200, 209):
		valid_url = True
	else:
		valid_url = False

	my_request = urllib2.urlopen(inputUrl)

	mime = my_request.info()['Content-type']
	if mime.endswith("png"):
		valid_url = True
		image_url = True
	elif mime.endswith("jpeg"):
		valid_url = True
		image_url = True
	else:
		valid_url = False
		image_url = False


	if not valid_url | image_url:
		# if the input is invalid then sending 401 response.
		response = make_response(json.dumps('Invalid...'), 401)
		response.headers['Content-Type'] = 'application/json'
		return render_template('index.html', name=inputUrl , comment = response , result = "Invalid Input")
	else:
		
		# Calling ML algo to check the validity
		DATA_TO_SEND , probability = MNIST_predictions(inputUrl)
		#print DATA_TO_SEND
		FinalResultObj = MNIST_classification(inputUrl , DATA_TO_SEND , probability )
		json_format =  jsonify(FinalResultObj.toJSON())
		return render_template('index.html', name=inputUrl , comment = json_format , result = str(DATA_TO_SEND))
	#comment = request.form['comment']
	#return 'Name is: ' + name + 'and the comment is:  ' + comment 
	#return render_template('index.html', name=name, comment=comment)
	#return render_template('index.html', name=inputUrl)


if __name__ == '__main__':
	app.run(debug=True)