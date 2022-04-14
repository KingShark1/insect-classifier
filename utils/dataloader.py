import os

from torchvision import datasets
import numpy

image_data_dir = 'data/detection/VOC2007/JPEGImages'
bbox_data_dir = 'data/detection/VOC2007/Annotations'
trainval_test_img = {'train': 'data/detection/VOC2007/ImageSets/Main/trainval.txt', 
'test': 'data/detection/VOC2007/ImageSets/Main/test.txt'}

def read_image_name(x):
	with open(trainval_test_img[x]) as f:
		filenames = f.readlines()
	return filenames

image_dataset = {x: read_image_name(x) for x in ['train', 'test']}
# print(image_dataset['train'][:5])

def load_image_and_bbox_paths():
	def read_bbox(image_name_list: list) -> list:
		box_list = []
		for box in image_name_list:
			box_list.append(os.path.join(bbox_data_dir, f"{box.strip()}.xml"))
		return box_list

	bbox_paths = {x: read_bbox(image_dataset[x]) for x in ['train', 'test']}
	# print("Bbox path for 0th image bbox : \t\t",bbox_paths['train'][0])

	def read_image_path(image_name_list: list) -> list:
		image_path_list = []
		for img in image_name_list:
			image_path_list.append(os.path.join(image_data_dir, f"{img.strip()}.jpg"))
		return image_path_list

	image_paths = {x: read_image_path(image_dataset[x]) for x in ['train', 'test']}
	print("Imgae path for 0 index : \t\t", image_paths['train'][0])

	return image_paths, bbox_paths