import torch
from PIL import Image
from utils.dataloader import InsectDataset
from utils.preprocess import applied_transforms

image_data_dir = 'data/detection/VOC2007/JPEGImages'
bbox_data_dir = 'data/detection/VOC2007/Annotations'
trainval_img_dir = 'data/detection/VOC2007/ImageSets/Main/trainval.txt' 
test_img_dir = 'data/detection/VOC2007/ImageSets/Main/test.txt'

def load_model():
	model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
	
	# Chaning the last layer of AlexNet to use with our dataset
	model.classifier[6] = torch.nn.Linear(4096, 102)
	model.eval()
	return model

def load_image(idx):
	"""
	Loads, processes and return transformed Image in PIL Image format
	also returns name of class and bboxes of the insect, in the original image (in that order)
	"""

	train_dataset = InsectDataset(trainval_img_dir, image_data_dir, bbox_data_dir)

	tvision_transform = train_dataset[idx]
	return tvision_transform
