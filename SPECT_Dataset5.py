import torch
from torch.utils.data import Dataset
import numpy as np
import random


def read_float32_binary(file_name):
    # Read binary file as numpy array
    data_np = np.fromfile(file_name, dtype=np.float32)

    # Convert numpy array to PyTorch tensor
    return data_np


class SPECT_Dataset5(Dataset):
    def __init__(self, input_prefix, input_suffix, label_prefix, label_suffix, num_sets_or_list, normalize_input, add_noise):
        self.input_prefix = input_prefix
        self.input_suffix = input_suffix
        self.label_prefix = label_prefix
        self.label_suffix = label_suffix
        self.normalize_input = normalize_input
        self.add_noise = add_noise
        self.expansion = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.the_data = []
        self.norm_max = 0

        if isinstance(num_sets_or_list, int):
            self.num_sets = num_sets_or_list
            sets_list = range(num_sets_or_list)
        elif isinstance(num_sets_or_list, list):
            sets_list = num_sets_or_list
            self.num_sets = len(sets_list)
        else:
            raise ValueError("Input must be an integer or a list.")

        for idx in sets_list:
            # Load input and label binary files for the given index
            in_name = self.input_prefix + str(idx) + self.input_suffix
            label_name = self.label_prefix + str(idx) + self.label_suffix
            input_data = read_float32_binary(in_name)
            label_data = read_float32_binary(label_name)

            # input_data
            padded_image = np.pad(input_data.reshape(120, 250), ((68, 68), (3, 3)), mode='constant')
            input_data = torch.tensor(padded_image, device=self.device).view(1, 256, 256)

            # label data
            padded_image = np.pad(label_data.reshape(250, 250), ((3, 3), (3, 3)), mode='constant')
            label_data = torch.tensor(padded_image, device=self.device).view(1, 256, 256)

            # normalize as needed
            if self.normalize_input:
                self.norm_max = max( torch.max(input_data), self.norm_max)

            # save it
            self.the_data.append( (input_data, label_data) )

    def expand_data(self, expansion):
        assert 120 % expansion == 0, f"Invalid expansion: {expansion}."
        self.expansion = expansion

    def __len__(self):
        return self.num_sets * self.expansion

    def __getitem__(self, idx_in):
        index = idx_in % self.num_sets
        assert 0 <= index <= self.num_sets
        x, y = self.the_data[index]

        # roll if necessary; x is cloned in this block
        if idx_in >= self.num_sets:
            step = 120 // self.expansion
            roll_index = idx_in//self.num_sets
            assert 1 <= roll_index < self.expansion
            x = self.roll_image(x, roll_index * step)
        else:
            # roll_image will clone
            x = x.clone()

        # clone y so it can be modified
        y = y.clone()

        # normalize
        if self.normalize_input:
            # add noise before normalizing
            if self.add_noise:
                x = torch.poisson(x)

            assert self.norm_max > 0
            x /= self.norm_max
            y /= self.norm_max
        else:
            # get a scale factor for the activity
            scale_factor = random.uniform(0.5, 3.0)
            x *= scale_factor
            y *= scale_factor

            # add noise after scaling
            if self.add_noise:
                x = torch.poisson(x)
 
        # return result
        return x, y
 

    def roll_image(self, x, roll_amount):
        assert 0 <= roll_amount < 120
        rolled_x = x.clone().squeeze()

        # Specify the portion of the tensor you want to roll
        slice_indices = (slice(68, 188), slice(0, 255))

        # Extract the portion of the tensor to roll
        portion_to_roll = rolled_x[slice_indices]

        # Roll the portion along the first dimension by 1 position
        rolled_portion = torch.roll(portion_to_roll, shifts=roll_amount, dims=0)

        # Replace the rolled portion in the copied tensor
        rolled_x[slice_indices] = rolled_portion

        # return result
        return  rolled_x.unsqueeze(0)

