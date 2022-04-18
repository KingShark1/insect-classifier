import torch

def load_model(device: torch.device) -> torch.nn.Module:
	"""
	Loads torchvision GoogleNet model with pretrained weights, converts the last layer of model to fit with our insect dataset with 102 classes
	and ports the model to given device (gpu/cpu), and returns it

	Parameters :
		device - torch device
	
	Returns :
		Finetuned GoogleNet model ported to GPU (if available)
	"""
	model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
	
	# Training as a fixed feature extractor
	for param in model.parameters():
		param.requires_grad = False
	
	# Changing the last layer of googlenet model
	num_ftrs = model.fc.in_features
	model.fc = torch.nn.Linear(num_ftrs, 102)
	model.to(device)

	return model