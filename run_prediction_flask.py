from flask import Flask, render_template, request, redirect, jsonify, url_for, flash
import httplib2
import json
import requests
app = Flask(__name__)

@app.route('/home')
def showHome():
    return render_template('YOUR_PAGE.html')
@app.route('/imageCheck', methods=['GET','POST'])
def checkImage():
if request.form['name']:
    #inputUrl is input received from HTML page.
    inputUrl = request.form['name']
     if INPUT IS INVALID THEN:
        # if the input is invalid then sending 401 response.
        response = make_response(json.dumps('Invalid...'), 401)
        response.headers['Content-Type'] = 'application/json'
        return response
    else:
        # Calling ML algo to check the validity
        DATA_TO_SEND=ml_function()
        return jsonify(Data=[DATA_TO_SEND])

if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
