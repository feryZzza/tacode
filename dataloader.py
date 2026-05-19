import os
from typing import List, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset


class TcnDataset(Dataset):
	'''Dataset for dynamically loading input and label data based on indices.'''
	def __init__(self,
					data_dir: str,
					input_names: List[str],
					label_names: List[str],
					side: str,
					participant_masses: Dict[str, float] = {},
					device: torch.device = torch.device("cpu")):
		self.data_dir = data_dir
		self.input_names = input_names
		self.label_names = label_names
		self.side = side
		self.participant_masses = participant_masses
		self.device = device
		self.trial_names = self._get_trial_names()

	def __len__(self):
		'''Returns number of files found.'''
		return len(self.trial_names)

	def __getitem__(self,
					idx: int or List[int] or slice):
		'''Loads data based on provided indices. Uses zero padding to concatenate trials of different size.'''
		# Get list of desired file names based on idx
		if isinstance(idx, list):
			trial_names = [self.trial_names[i] for i in idx]
		else:
			trial_names = self.trial_names[idx]
			trial_names = [trial_names] if not isinstance(trial_names, list) else trial_names

		# Load data
		data = [list(self._load_trial_data(trial_name)) for trial_name in trial_names]

		# add zero padding to allow for concatenation
		data, trial_sequence_lengths = self._add_zero_padding(data)
		
		# concatenate tensors
		input_data, label_data = zip(*data)
		input_data = torch.cat(input_data, dim = 0)
		label_data = torch.cat(label_data, dim = 0)

		return input_data, label_data, trial_sequence_lengths

	def get_trial_names(self):
		return self.trial_names

	def _get_trial_names(self):
		'''Get all trial names in data_dir.'''
		# extract participant directories
		participants = [participant for participant in os.listdir(self.data_dir) if "." not in participant and participant != "LICENSE"]

		# iterate through participant directories and get trial names
		trial_names = []
		for participant in participants:
			participant_dir = os.path.join(self.data_dir, participant)
			for trial_name in os.listdir(participant_dir):
				trial_names.append(os.path.join(participant, trial_name))

		return trial_names

	def _load_trial_data(self, trial_name: str):
		'''Loads data from a single trial.'''
		print(f"Loading {trial_name}.")
		
		# load input data
		input_file_path = os.path.join(self.data_dir, trial_name, "Exo.csv")
		participant = trial_name.split("/")[0].split("\\")[0] # get participant name for body mass normalization
		if participant not in self.participant_masses:
			print(f"Warning - {participant} mass was not provided.")
		input_data = self._load_input_data(input_file_path, body_mass = self.participant_masses.get(participant, 1.))

		# load label data
		label_file_path = os.path.join(self.data_dir, trial_name, "Joint_Moments_Filt.csv")
		label_data = self._load_label_data(label_file_path)

		return input_data, label_data

	def _load_input_data(self, file_path: str, body_mass: float):
		'''Loads input data from a single file and returns as a 3D torch.FloatTensor.'''

		# load as DataFrame
		df = pd.read_csv(file_path)

		# normalize pressure insole data by body mass
		df.loc[:, "insole_l_force_y"] /= body_mass
		df.loc[:, "insole_r_force_y"] /= body_mass

		# if left leg data, mirror sensors
		if self.side == "l":
			df.loc[:, "foot_imu_l_gyro_x"] *= -1.
			df.loc[:, "foot_imu_l_gyro_y"] *= -1.
			df.loc[:, "foot_imu_l_accel_z"] *= -1.
			df.loc[:, "shank_imu_l_gyro_x"] *= -1.
			df.loc[:, "shank_imu_l_gyro_y"] *= -1.
			df.loc[:, "shank_imu_l_accel_z"] *= -1.
			df.loc[:, "thigh_imu_l_gyro_x"] *= -1.
			df.loc[:, "thigh_imu_l_gyro_y"] *= -1.
			df.loc[:, "thigh_imu_l_accel_z"] *= -1.
			df.loc[:, "insole_l_cop_z"] *= -1.

		# convert to input and label tensors
		input_data = torch.tensor(df[self.input_names].values, device = self.device).transpose(0, 1).unsqueeze(0).float()

		return input_data

	def _load_label_data(self, file_path: str):
		'''Loads label data from a single file and returns as a 3D torch.FloatTensor.'''

		# load as DataFrame
		df = pd.read_csv(file_path)

		# convert to input and label tensors
		label_data = torch.tensor(df[self.label_names].values, device = self.device).transpose(0, 1).unsqueeze(0).float()

		return label_data

	def _add_zero_padding(self, 
							data: List[List[torch.FloatTensor]]):
		'''Adds zero padding to the end of each trial to match the sequence lengths of all trial data.'''
		trial_sequence_lengths = [trial_data[0].shape[-1] for trial_data in data]
		max_sequence_length = max(trial_sequence_lengths)

		# iterate through each trial and add zero padding as needed
		for i in range(len(data)):
			trial_sequence_length = trial_sequence_lengths[i]
			if trial_sequence_length < max_sequence_length:
				# pad input data and label data
				padding_length = max_sequence_length - trial_sequence_length
				for j in range(len(data[i])):
					data[i][j] = torch.cat((data[i][j], torch.zeros((1, data[i][j].shape[1], padding_length), device = self.device)), dim = 2)

		return data, trial_sequence_lengths
