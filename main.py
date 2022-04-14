# import cv2
from torchvision import transforms
import torch

from utils import display
from utils.preprocess import applied_transforms, load_image, read_content

from models.alexnet import AlexNet

def main():
	input_tensor, name, bbox = AlexNet.load_image()
	input_batch = input_tensor.unsqueeze(0)
	tensor_to_img = transforms.ToPILImage()
	
	tvision_transform = tensor_to_img(input_tensor)
	display.show_image(tvision_transform)
	

	model = AlexNet.load_model()

	if torch.cuda.is_available():
		input_batch = input_batch.to('cuda')
		model.to('cuda')

	with torch.no_grad():
		output = model(input_batch)
	
	print(output[0].size())

	probabilities = torch.nn.functional.softmax(output[0], dim=0)
	print(probabilities.size())
	"""
	image_path = ['./data/detection/VOC2007/JPEGImages/IP000000003.jpg', 'data/detection/VOC2007/JPEGImages/IP102005722.jpg']
	image_crop = ['./data/detection/VOC2007/Annotations/IP000000003.xml', 'data/detection/VOC2007/Annotations/IP102005722.xml']
	
	image = load_image(image_path[0])
	# gray_image = color_to_gray(image)
	# cannyImg = canny_detection(gray_image)
	
	name, bbox = read_content(image_crop[0])
	
	print(f"{name}\nbounding boxes : {bbox}")

	# insects = crop(cannyImg, bbox)
	# for i in insects:
	# 	i = resize(i)
	# 	display.show_image(i)
	tensor_to_img = transforms.ToPILImage()
	
	tvision_transform = tensor_to_img(applied_transforms(image, bbox))
	display.show_image(tvision_transform)
	"""

if __name__=="__main__":
	main()