from flask import Flask, render_template, request, make_response
from functools import wraps, update_wrapper
from PIL import Image
from predictor import Predictor

import numpy as np
import torch

# Initialize the predictor object.
predictor = Predictor()

# Initialize Flask app.
app = Flask(__name__)

# Define function to disable caching in Flask.
def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)

# Base endpoint to perform prediction.
@app.route('/', methods=['GET', 'POST'])
@nocache
def upload():
    if request.method == 'POST':
        prediction = predictor.predict(request)
        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html', prediction=None)
		
if __name__ == '__main__':
   app.run()
