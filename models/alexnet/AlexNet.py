import torch
from PIL import Image
from utils.preprocess import applied_transforms
from utils.dataloader import load_image_and_bbox_paths
def load_model():
	model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
	
	# Chaning the last layer of AlexNet to use with our dataset
	model.classifier[6] = torch.nn.Linear(4096, 102)
	model.eval()
	return model

def load_image():
	"""
	Loads, processes and return transformed Image in PIL Image format
	also returns name of class and bboxes of the insect, in the original image (in that order)
	"""

	image_path, bbox_path = load_image_and_bbox_paths()

	tvision_transform, name, bbox = applied_transforms(image_path['train'][0], bbox_path['train'][0])
	return tvision_transform, name, bbox

