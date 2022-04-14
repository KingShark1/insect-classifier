from PIL import Image, ImageDraw

def show_image(image, bbox=None, name=None):
	image.show()

def show_image_with_bbox(image, bbox, name):
	draw = ImageDraw.Draw(image)
	for b in bbox:
		draw.rectangle(((b[0], b[1]), (b[2], b[3])), fill=None, outline=255)
	
	draw.text((20,70), name)
	image.show()

	