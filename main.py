import torch

from utils import display, train
from utils import dataloader
from utils.dataloader import InsectDataset, read_image_name, show_insect_image
from utils.preprocess import read_content

from models.alexnet import AlexNet
from models.googlenet import GoogleNet
from models.basic_rnn import my_rnn

def load_criterion_optimizer_scheduler(model):
	criterion = torch.nn.CrossEntropyLoss()

	# All parameters are being optimized
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	# Decay LR by factor of 0.1 every 7 epochs
	exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	return criterion, optimizer, exp_lr_scheduler

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
	
	# alex_net = AlexNet.load_model(device=device)
	# criterion, optimizer, exp_lr_scheduler = load_criterion_optimizer_scheduler(alex_net)
	# train.train_model(alex_net, 
	# 												dataloaders=dataloaders,
	# 												dataset_sizes=dataset_sizes,
	# 												device=device,
	# 												criterion=criterion,
	# 												optimizer=optimizer,
	# 												scheduler=exp_lr_scheduler)

	google_net= GoogleNet.load_model(device=device)
	criterion, optimizer, exp_lr_scheduler = load_criterion_optimizer_scheduler(google_net)
	# Training the model
	model = train.train_model(google_net, 
														dataloaders=dataloaders,
														dataset_sizes=dataset_sizes,
														device=device,
														criterion=criterion,
														optimizer=optimizer,
														scheduler=exp_lr_scheduler)
	
	
	# my_RNN = my_rnn.load_model(device=device)
	# criterion, optimizer, exp_lr_scheduler = load_criterion_optimizer_scheduler(my_RNN)
	# # Training the model
	# train.train_model(my_RNN,
	# 											dataloaders=dataloaders,
	# 											dataset_sizes=dataset_sizes,
	# 											device=device,
	# 											criterion=criterion,
	# 											optimizer=optimizer,
	# 											scheduler=exp_lr_scheduler)

if __name__=="__main__":
	main()