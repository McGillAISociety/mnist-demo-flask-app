from nn import Net  

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF

save_model_path = './results/model.pth'
save_optimizer_path = './results/optimizer.pth'

size = (28, 28)

def load_data(batch_size_train=64, batch_size_test=1000):
	'''
	This method loads the MNIST dataset from Pytorch. It takes care of the normalization
	and pretraining transformation. 
	'''

	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST('../data', train=True, download=True,
									transform=torchvision.transforms.Compose([
										torchvision.transforms.ToTensor(),
										torchvision.transforms.Normalize(
											(0.1307,), (0.3081,))
									])),
		batch_size=batch_size_train, shuffle=True)

	test_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST('../data', train=False, download=True,
									transform=torchvision.transforms.Compose([
										torchvision.transforms.ToTensor(),
										torchvision.transforms.Normalize(
											(0.1307,), (0.3081,))
									])),
		batch_size=batch_size_test, shuffle=True)

	return train_loader, test_loader

def train(model, train_data, optimizer, log_interval=100): 
	'''
	This method trains the neural network. 
	The model is saved after each epoch in `./results/` 
	'''
	model.train()
	for batch_idx, (data, target) in enumerate(train_data):
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_data.dataset),
				100. * batch_idx / len(train_data), loss.item()))
			train_losses.append(loss.item())
			train_counter.append(
				(batch_idx*64) + ((epoch-1)*len(train_data.dataset)))
			torch.save(model.state_dict(), save_model_path)
			torch.save(optimizer.state_dict(), save_optimizer_path)

def test(model, test_data):
	'''
	This method evaluates the train model and outputs the final accuracy results. 
	'''
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_data:
			output = model(data)
			test_loss += F.nll_loss(output, target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(test_data.dataset)
	test_losses.append(test_loss)
	print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_data.dataset),
		100. * correct / len(test_data.dataset)))


def predict(sample, model_path='./results/model.pth'): 
	'''
	This method makes a prediction on a single image, and returns the result. 
	'''
	model = Net()
	model.load_state_dict(torch.load(model_path))
	output = model(sample)
	return output 

if __name__ == "__main__" :
	# Init the network and the optimizer. 
	network = Net()
	
	learning_rate = 0.01
	momentum = 0.5
	optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

	# Define model training parameters. 
	n_epochs = 3
	random_seed = 1
	torch.manual_seed(random_seed)

	# Load the data. 
	train_loader, test_loader = load_data()

	examples = enumerate(test_loader)
	batch_idx, (example_data, example_targets) = next(examples)

	print(example_data.shape)

	# Init training and testing logs.
	train_losses = []
	train_counter = []
	test_losses = []
	test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

	for epoch in range(1, n_epochs + 1):
		train(network, train_loader, optimizer)
		test(network, test_loader)
	
	
	
	

