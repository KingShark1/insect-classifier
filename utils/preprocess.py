"""
Authoured By - Manas Tiwari
Email - manastiwari28@gmail.com
"""


from PIL import Image, ImageOps, ImageFilter
import xml.etree.ElementTree as ET

import torchvision.transforms as transforms
import torch

def load_image(path_to_image: str):
	"""
	Parameters 
		path_to_image : relative path to where image is stored
	
	Returns
		Image object
	"""
	return Image.open(path_to_image)

def read_content(xml_file: str):
	"""
	Read contents of the xml file and returns bounding boxes in a nested list

	Parameters : 
		relative path of xml file

	Returns : 
		Name of the file
		Nested List (2D List)
	"""
	tree = ET.parse(xml_file)
	root = tree.getroot()

	list_with_all_boxes = []

	for boxes in root.iter('object'):
		filename = root.find('filename').text

		xmin = int(boxes.find('bndbox/xmin').text)
		ymin = int(boxes.find('bndbox/ymin').text)
		xmax = int(boxes.find('bndbox/xmax').text)
		ymax = int(boxes.find('bndbox/ymax').text)

		list_with_single_boxes = [xmin, ymin, xmax, ymax]
		list_with_all_boxes.append(list_with_single_boxes)
	
	return int(filename[2:5]), list_with_all_boxes



# def color_to_gray(image):
# 	"""
# 	converts Image object to grayscale Image object

# 	Parameters
# 		image : Image in PIL Image format

# 	Returns
# 		grayscale image
# 	"""
# 	return ImageOps.grayscale(image)

# def canny_detection(grayscale_image):
#   """
#   Applies canny edge detection as in PIL ImageFilter

#   Patameters :
# 		grayscale Image

#   Returns : 
# 		Image transformed with canny edge detection
#   """
#   return grayscale_image.filter(ImageFilter.FIND_EDGES)

# def crop(image, bbox):
# 	"""
# 	Applies crop according to bounding box

# 	Parameters : 
# 		image : PIL Image object
# 		bbox : bounding boxes, as given in annotations xml, required as 2D integer List

# 	Returns : List with PIL Image objects
# 	"""
# 	insects = []
# 	for insect_box in bbox:
# 		left, top, right, bottom = insect_box[0], insect_box[1], insect_box[2], insect_box[3]
# 		insects.append(image.crop((left+1, top+1, right, bottom)))
# 	return insects

# def resize(image, size=(227, 227)):
# 	"""
# 	Resizes the image while keeping the aspect ration constant

# 	Parameters :
# 		image : PIL Image object
# 		size : size of the output image, defaults to 227,227
# 	"""
# 	return image.resize(size, Image.ANTIALIAS)

class MyCrop:
	"""Crop by bounding boxes"""

	def __init__(self, bbox):
		self.bbox = bbox
	
	def __call__(self, x):
		insects = []
		for insect_box in self.bbox:
			left, top, right, bottom = insect_box[0], insect_box[1], insect_box[2], insect_box[3]
			insects.append(x.crop((left+1, top+1, right, bottom)))
		return insects[0]

class MyCanny:
	"""Canny Edge Detection"""

	def __init__(self) -> None:
			pass
	
	def __call__(self, x) -> Image:
		return x.filter(ImageFilter.FIND_EDGES)


def applied_transforms(path_to_image: str, path_to_bbox:str) -> Image:
	"""
	Compiles transforms and returns image in *Tensor* format
	"""
	image = load_image(path_to_image=path_to_image)
	name, bbox = read_content(path_to_bbox)

	transform = transforms.Compose([
		transforms.Grayscale(3),
		MyCanny(),
		MyCrop(bbox),
		transforms.Resize((227, 227)),		
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(20),
		transforms.RandomVerticalFlip(p=0.3),
		transforms.ToTensor(),
		])

	return transform(image), name, bbox