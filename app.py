#uses same template as boston housing with categorical variables encoded numbers coming from the web page

from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import pickle 
import joblib
import json
import numpy as np
import requests
import request
import pandas as pd
from flask_cors import CORS

#https://www.kdnuggets.com/2019/10/easily-deploy-machine-learning-models-using-flask.html

app = Flask(__name__)
cors = CORS(app)

def create_app():
    model = joblib.load(open('airbnb_model.sav', 'rb'))

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
