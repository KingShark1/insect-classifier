import copy
import time
import torch


def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler, num_epochs:int=10):
	"""
	Parameters :
		model - the model to be trained
	"""
	
	
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print(f'Epoch {epoch} / {num_epochs - 1}')
		print('-' * 10)

		# Each Epoch has a training and a validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				# Set model to training mode
				model.train()
			else:
				# Set model to evaluation mode
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# Forward Pass
				# Track history if only in training phase
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# Backward pass + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()
				
				# Statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			
			if phase == 'train':
				scheduler.step()
			
			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc: .4f}')

			# Deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()
	
	time_elapsed = time.time() - since
	print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
	print(f'Best val Acc: {best_acc:4f}')

	# Load best model weights
	model.load_state_dict(best_model_wts)
	return model