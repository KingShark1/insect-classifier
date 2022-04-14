import torch
from PIL import Image
from utils.preprocess import applied_transforms, load_image, read_content

def load_model():
	model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
	model.eval()
	return model

def load_image():
	"""
	Loads, processes and return transformed Image in PIL Image format
	also returns name of class and bboxes of the insect, in the original image (in that order)
	"""
	image_path = ['./data/detection/VOC2007/JPEGImages/IP000000003.jpg', 'data/detection/VOC2007/JPEGImages/IP102005722.jpg']
	image_crop = ['./data/detection/VOC2007/Annotations/IP000000003.xml', 'data/detection/VOC2007/Annotations/IP102005722.xml']
	
	image = Image.open(image_path[0])
	name, bbox = read_content(image_crop[0])
	tvision_transform = applied_transforms(image, bbox)
	return tvision_transform, name, bbox

