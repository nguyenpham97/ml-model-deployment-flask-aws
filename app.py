# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:00:02 2021

@author: nguyen
"""

from flask import Flask, jsonify, request
import joblib

import model as model_file

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('model.pkl')
    count_vect = joblib.load('count_vect.pkl')
    to_predict_list = request.form.to_dict()
    review_text = model_file.clean_text(to_predict_list['review_text'])
    pred = clf.predict(count_vect.transform([review_text]))
    if pred[0]:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return flask.render_template('index.html', prediction_text = f"The review is: {prediction}") #jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)