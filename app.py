from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import flask
import pickle 
import json
import numpy as np
import pandas as pd
from flask_cors import CORS


import traceback

#https://hackernoon.com/machine-learning-w22g322x: start March 26

app = Flask(__name__,template_folder='templates')
cors = CORS(app)

def create_app():
    with open('airbnb_model.pkl', 'rb') as f:
        model = pickle.load(f)
 
    with open('airbnb_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)

#################APP ROUTES####################

    @app.route('/')
    def welcome():
        return "AirBNB Listing Price Prediction: API by Bhavneesh Sharma"

    @app.route('/predict', methods=['POST','GET'])
    def predict():
  
        if flask.request.method == 'GET':
            return "Prediction page"
 
        if flask.request.method == 'POST':
            try:
                json_ = request.get_json()
                print(json_)
                #query_ = pd.get_dummies(pd.DataFrame(json_))
                query_ = pd.DataFrame(json_)
                query = query_.reindex(columns = model_columns, fill_value= 0)
                prediction = list(model.predict(query))
 
                return jsonify({
                    "prediction":str(prediction)
                })
 
            except:
                return jsonify({
                    "trace": traceback.format_exc()
                    })
      
    return app
if __name__ == "__main__":
    app.run(debug=True)