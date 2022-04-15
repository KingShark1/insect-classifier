import os
from random import sample

from torch.utils.data import Dataset
import torch

from utils.preprocess import applied_transforms

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


class InsectDataset(Dataset):
	"""
	Insect Dataset
	"""
	def __init__(self, txt_file_path: list, image_dir: str, bbox_dir: str):
		"""
		txt_file_list - trainval or test file contents loaded as list from ImageSets/Main/test.txt or trainval.txt
		image_dir - Path to JPEGimages
		bbox_dir - Path to annotations
		"""
		self.txt_file = read_image_name(txt_file_path)
		self.bbox_dir = bbox_dir
		self.image_dir = image_dir

	def __len__(self):
		return len(self.txt_file)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		img_path = os.path.join(self.image_dir, self.txt_file[idx] + '.jpg')
		bbox_path = os.path.join(self.bbox_dir, self.txt_file[idx] + '.xml')
		
		transform, _name, _bbox = applied_transforms(img_path, bbox_path)

		return transform

		


	