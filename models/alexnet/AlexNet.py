import torch

def load_model(device: torch.device) -> torch.nn.Module:
	"""
	Loads torchvision Alexnet model with pretrained weights, converts the last layer of model to fit with our insect dataset with 102 classes
	and ports the model to given device (gpu/cpu), and returns it

	Parameters :
		device - torch device
	
	Returns :
		Finetuned Alexnet model ported to GPU (if available)
	"""
	model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
	
	# Chaning the last layer of AlexNet to use with our dataset
	model.classifier[6] = torch.nn.Linear(4096, 102)

	model.to(device)

	return model