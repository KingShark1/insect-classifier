from torch.autograd import Variable
import torch


import torch.nn as nn

class ImageRNN(nn.Module):

	def __init__(self, input_dim=227, hidden_dim=1000, layer_dim=1, output_dim=102) -> None:
		super(ImageRNN, self).__init__()

		self.hidden_dim = hidden_dim
		self.layer_dim = layer_dim

		self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		
		# Initialize the hidden state with zeros
		h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

		print(f'x shape :{x.size()} \t h0 shape:{h0.size()}')
		out, hn = self.rnn(x, h0) 
		out = self.fc(out[:, -1, :])
		return out
	

def load_model(device: torch.device) -> torch.nn.Module:
	model = ImageRNN()
	model.to(device)
	return model

