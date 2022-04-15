# import cv2
from torchvision import transforms
import torch

from utils import display
from utils.dataloader import read_image_name
from utils.preprocess import applied_transforms, load_image, read_content

from models.alexnet import AlexNet

image_data_dir = 'data/detection/VOC2007/JPEGImages'
bbox_data_dir = 'data/detection/VOC2007/Annotations'
trainval_img_dir = 'data/detection/VOC2007/ImageSets/Main/trainval.txt' 
test_img_dir = 'data/detection/VOC2007/ImageSets/Main/test.txt'

def main():
	input_tensor = AlexNet.load_image(15)
	input_batch = input_tensor.unsqueeze(0)
	tensor_to_img = transforms.ToPILImage()
	
	tvision_transform = tensor_to_img(input_tensor)
	display.show_image(tvision_transform)
	

	# model = AlexNet.load_model()

	# if torch.cuda.is_available():
	# 	input_batch = input_batch.to('cuda')
	# 	model.to('cuda')

	# with torch.no_grad():
	# 	output = model(input_batch)

	train_val_imageset = read_image_name(trainval_img_dir)
	print(train_val_imageset[0])
	

if __name__=="__main__":
	main()