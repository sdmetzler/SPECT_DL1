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
    def __init__(self, input_prefix, input_suffix, label_prefix, label_suffix, num_sets_or_list, normalize_input, normalize_label, 
            add_noise):
        self.input_prefix = input_prefix
        self.input_suffix = input_suffix
        self.label_prefix = label_prefix
        self.label_suffix = label_suffix
        self.normalize_input = normalize_input
        self.normalize_label = normalize_label
        self.add_noise = add_noise
        self.expansion = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.the_data = []

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
                input_data /= torch.max(input_data)

            if self.normalize_label:
                label_data /= torch.max(label_data)

            # save it
            self.the_data.append( (input_data, label_data) )

    def expand_data(self, expansion):
        assert 120 % expansion == 0, f"Invalid expansion: {expansion}."
        self.expansion = expansion

    def __len__(self):
        return self.num_sets * self.expansion

    def __getitem__(self, idx_in):
        index = idx_in % self.num_sets
        x, y = self.the_data[index]

        if idx_in >= self.num_sets:
            step = 120 // self.expansion
            roll = step * (idx_in//self.num_sets)
            assert roll >= step
            x = self.roll_image(x.clone(), roll)

            

        # scale
        """
        This is commented out for now until the data is fixed. For now,
        I won't normalize.
        if (not self.normalize_input) or (not self.normalize_label):
            # get a scale factor for the activity
            scale_factor = random.uniform(0.5, 3.0)
            if not self.normalize_input:
                x *= scale_factor
            if not self.normalize_label:
                y *= scale_factor
        """
 
        if self.add_noise:
            # generate Poisson noise
            #poisson_scale = 180_000 * scale_factor
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

