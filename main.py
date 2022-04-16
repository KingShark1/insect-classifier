import torch

from utils import display, train
from utils import dataloader
from utils.dataloader import InsectDataset, read_image_name, show_insect_image
from models.alexnet import AlexNet
from utils.preprocess import read_content


def main():
	# Shows insect image
	# show_insect_image(14550)
	
	datasets = {'train': InsectDataset(txt_file_path='data/detection/VOC2007/ImageSets/Main/trainval.txt'),
							'val': InsectDataset(txt_file_path='data/detection/VOC2007/ImageSets/Main/test.txt')}
	dataloaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=4, shuffle=True, num_workers=4),
								'val': torch.utils.data.DataLoader(datasets['val'], batch_size=4, shuffle=True, num_workers=4)}
	dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
	class_names = datasets['train'].classes

	
	# device available
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	model = AlexNet.load_model(device=device)
	criterion, optimizer, exp_lr_scheduler = AlexNet.load_criterion_optimizer_scheduler(model)
	
	# Training the model
	model = train.train_model(model, 
														dataloaders=dataloaders,
														dataset_sizes=dataset_sizes,
														device=device,
														criterion=criterion,
														optimizer=optimizer,
														scheduler=exp_lr_scheduler)
	
	
if __name__=="__main__":
	main()