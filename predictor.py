import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

from model.model import MNISTModel
from PIL import Image

size = (28, 28)

def preprocess_img(image):
	'''
	This method processes the image into the correct expected shape in the model (28, 28). 
	''' 
	if (image.mode == 'RGB'): 
		# Convert RGB to grayscale. 
		image = image.convert('L')
	image = image.resize(size)
	return image

def image_loader(image):
	''' 
	This method loads the image into a PyTorch tensor. 
	'''
	image = TF.to_tensor(image)
	image = image.unsqueeze(0)
	return image

class Predictor: 
	def __init__(self):
		# ======== YOUR CODE ========= #

	def predict(self, request):
		'''
		This method reads the file uploaded from the Flask application POST request, 
		and performs a prediction using the MNIST model. 
		'''
		# ======== YOUR CODE ========= #
	