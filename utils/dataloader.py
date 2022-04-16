import os
from random import sample


from torch.utils.data import Dataset
import torch

import torchvision.transforms as transforms
from utils.preprocess import applied_transforms
from utils import relative_paths
from utils import display

image_data_dir = 'data/detection/VOC2007/JPEGImages'
bbox_data_dir = 'data/detection/VOC2007/Annotations'
trainval_img_dir = 'data/detection/VOC2007/ImageSets/Main/trainval.txt' 
test_img_dir = 'data/detection/VOC2007/ImageSets/Main/test.txt'

def read_image_name(img_dir):
	"""
	Do this only one time and save the names, helps in speeding up runtime in dataloading
	"""
	with open(img_dir) as f:
		filenames = f.readlines()
	
	for file in range(len(filenames)):
		filenames[file] = filenames[file].strip()
	return filenames

def convert_class_to_dictionary(classes_path: str) -> dict:
	"""
	classes_path - Relative Path to classes.txt file, containing the classes of insects

	returns - Dictionary conatiaing key as integers and values as str of scienctific name of the corresponding insect
	"""
	
	classes = {}
	file = open(classes_path)
	for line in file:
		key = int(line.split()[0])
		value = ' '.join(line.split()[1:])
		classes[key] = value

	return classes

def show_insect_image(idx):
	"""
	Loads, processes and return transformed Image in PIL Image format
	also returns name of class and bboxes of the insect, in the original image (in that order)
	"""

	train_dataset = InsectDataset(relative_paths.trainval_img_dir, relative_paths.image_data_dir, relative_paths.bbox_data_dir)
	to_pil = transforms.ToPILImage()

	transforms_on_idx, name = train_dataset[idx]
	print(name)
	display.show_image(to_pil(transforms_on_idx), name=name)
	

class InsectDataset(Dataset):
	"""
	Insect Dataset
	"""
	def __init__(self, txt_file_path: str, 
							image_dir: str='data/detection/VOC2007/JPEGImages', 
							bbox_dir: str='data/detection/VOC2007/Annotations', 
							classes: str='data/classes.txt'):
		"""
		txt_file_list - trainval or test file contents loaded as list from ImageSets/Main/test.txt or trainval.txt
		image_dir - Path to JPEGimages
		bbox_dir - Path to annotations
		"""
		self.txt_file = read_image_name(txt_file_path)
		self.bbox_dir = bbox_dir
		self.image_dir = image_dir
		self.classes = convert_class_to_dictionary(classes)

	def __len__(self):
		return len(self.classes)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		img_path = os.path.join(self.image_dir, self.txt_file[idx] + '.jpg')
		bbox_path = os.path.join(self.bbox_dir, self.txt_file[idx] + '.xml')
		
		transform, name, _bbox = applied_transforms(img_path, bbox_path)

		return transform, name

		


	