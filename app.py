#uses same template as boston housing with categorical variables encoded numbers coming from the web page

from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import pickle 
import json
import numpy as np
import requests
import request
import pandas as pd
from flask_cors import CORS

import traceback

#https://hackernoon.com/machine-learning-w22g322x: start March 26

app = Flask(__name__,template_folder='templates')
cors = CORS(app)

def create_app():
    with open('airbnb_model.pkl', 'rb') as f:
    model = pickle.load(f)
 
    with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

#################APP ROUTES####################

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/predict', methods=['POST', 'GET'])
    def predict():
        model = joblib.load(open('airbnb_model.sav', 'rb'))
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0])
        #return render_template('index.html', prediction_text='The predicted home price is = $ {}'.format(output))
        return jsonify(output)
    
    @app.route('/api', methods=['POST', 'GET'])
    def prediction():
        data = request.get_json(force=True)
        model = joblib.load(open('airbnb_model.sav', 'rb'))
        prediction = model.predict([[np.array(data['room_type'], ['neighbourhood_group'])]])
        output = round(prediction[0])
        return jsonify(output)
    return app

if __name__ == "__main__":
    model = joblib.load(open('airbnb_model.sav', 'rb'))
    app.run(debug=True)
