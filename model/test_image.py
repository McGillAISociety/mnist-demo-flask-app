import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

from main import predict
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

def test_single_image(image_path): 
	'''
	This method tests the model using a single image. 
	'''
	test_img = Image.open(image_path)
	
	# First put the image into the right format. 
	sample = preprocess_img(test_img)
	sample = image_loader(sample)

	# Predict the output. 
	prediction_tensor = predict(sample)
	prediction_label = torch.argmax(prediction_tensor)
	print('The model recognizes this picture as a {}'.format(prediction_label))
	
	# Show the test image. 
	plt.imshow(test_img)
	plt.show()

if __name__ == '__main__' :
    test_single_image('./test.jpg')
	