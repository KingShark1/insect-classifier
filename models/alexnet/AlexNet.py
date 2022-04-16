import torch

def load_model(device: torch.device) -> torch.nn.Module:
	"""
	Loads torchvision Alexnet model with pretrained weights, converts the last layer of model to fit with our insect dataset with 102 classes
	and ports the model to given device (gpu/cpu), and returns it

	Parameters :
		device - torch device
	
	Returns :
		Finetuned Alexnet model in eval mode
	"""
	model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
	
	# Chaning the last layer of AlexNet to use with our dataset
	model.classifier[6] = torch.nn.Linear(4096, 102)

	model.to(device)

	return model

def load_criterion_optimizer_scheduler(model):
	criterion = torch.nn.CrossEntropyLoss()

	# All parameters are being optimized
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	# Decay LR by factor of 0.1 every 7 epochs
	exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	return criterion, optimizer, exp_lr_scheduler