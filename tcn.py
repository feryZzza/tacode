'''
This code implements the temporal convolutional network (TCN) class used in the study "Task-Agnostic Exoskeleton Control via Biological Joint Moment Estimation."

This code was modified from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py.
Original License: MIT License
Copyright (c) 2018 CMU Locus Lab
'''

from typing import List
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, dropout_type='Dropout', activation='ReLU', norm='weight_norm'):
		super(TemporalBlock, self).__init__()

		self.chomp1 = Chomp1d(padding)
		self.af1 = getattr(nn, activation)()
		self.dropout1 = getattr(nn, dropout_type)(dropout)

		self.chomp2 = Chomp1d(padding)
		self.af2 = getattr(nn, activation)()
		self.dropout2 = getattr(nn, dropout_type)(dropout)

		if norm == 'weight_norm':
			self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
				stride=stride, padding=padding, dilation=dilation))
			self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
				stride=stride, padding=padding, dilation=dilation))
			self.net = nn.Sequential(self.conv1, self.chomp1, self.af1, self.dropout1,
				self.conv2, self.chomp2, self.af2, self.dropout2)
		else:
			self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
				stride=stride, padding=padding, dilation=dilation)
			self.norm1 = getattr(nn, norm)(n_outputs)

			self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
				stride=stride, padding=padding, dilation=dilation)
			self.norm2 = getattr(nn, norm)(n_outputs)

			self.net = nn.Sequential(self.conv1, self.norm1, self.chomp1, self.af1, self.dropout1,
				self.conv2, self.norm2, self.chomp2, self.af2, self.dropout2)

		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None 
		self.af = getattr(nn, activation)()
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)		
		return self.af(out + res)


class TemporalConvNet(nn.Module):
	def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dropout_type='Dropout', activation='ReLU', norm='weight_norm'):
		super(TemporalConvNet, self).__init__()
		layers = []
		num_levels = len(num_channels)
		for i in range(num_levels):
			dilation_size = 2 ** i 
			in_channels = num_inputs if i == 0 else num_channels[i-1]
			out_channels = num_channels[i]
			layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, 
				padding=(kernel_size-1) * dilation_size, dropout=dropout, dropout_type=dropout_type,
				activation=activation, norm='weight_norm')]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)


class TCN(nn.Module):
	'''Implements the temporal convolutional network used in this study.'''
	def __init__(self, 
					input_size: int, 
					output_size: int, 
					num_channels: List[int], 
					ksize: int, 
					dropout: float, 
					eff_hist: int, 
					spatial_dropout: bool = False, 
					activation: str = 'ReLU', 
					norm: str = 'weight_norm', 
					center: float = 0., 
					scale: float = 1.):
		super(TCN, self).__init__()

		# create and initialize network
		self.dropout_type = 'Dropout2d' if spatial_dropout else 'Dropout'
		self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=ksize, dropout=dropout, dropout_type=self.dropout_type, activation=activation, norm=norm)
		self.linear = nn.Linear(num_channels[-1], output_size)
		self.init_weights()
		self.eff_hist = eff_hist

		# save for input feature normalization
		self.center = center
		self.scale = scale
		
	def init_weights(self):
		self.linear.weight.data.normal_(0, 0.01)

	def forward(self, x):
		# normalize input features
		out = (x - self.center) / self.scale

		# forward pass of conv layers
		out = self.tcn(out)

		# reshape for final linear layer
		out = torch.cat([out[i, :, :] for i in range(out.shape[0])], dim = 1).transpose(0, 1).contiguous()

		# forward pass of final linear layer
		out = self.linear(out).transpose(0, 1)

		# reshape back to original format
		out = torch.cat([out[:, i*x.shape[2]:(i+1)*x.shape[2]].unsqueeze(0) for i in range(x.shape[0])], dim = 0)

		return out

	def get_effective_history(self):
		return self.eff_hist