# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:57:29 2022

@author: 14198
"""

import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd


app = Flask(__name__)
#Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')


#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]#Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Percent with heart disease is {}'.format(output))

if __name__ == "__main__":
    app.run()