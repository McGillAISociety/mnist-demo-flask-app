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

# Base endpoint to perform prediction.
@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        prediction = predictor.predict(request)
        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html', prediction=None)
		
if __name__ == '__main__':
   app.run()
