import torch
from torch.utils.data import Dataset
import numpy as np


def read_float32_binary(file_name):
    # Read binary file as numpy array
    data_np = np.fromfile(file_name, dtype=np.float32)

    # Convert numpy array to PyTorch tensor
    return data_np


class SPECT_Dataset4(Dataset):
    def __init__(self, input_prefix, input_suffix, label_prefix, label_suffix, num_sets, expansion, normalize_input, normalize_label):
        self.input_prefix = input_prefix
        self.input_suffix = input_suffix
        self.label_prefix = label_prefix
        self.label_suffix = label_suffix
        self.num_sets = num_sets
        self.expansion = expansion
        assert 120 % expansion == 0, "Invalid expansion factor"
        self.normalize_input = normalize_input
        self.normalize_label = normalize_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.the_data = []
        for idx in range(num_sets):
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

            # save it
            self.the_data.append( (input_data, label_data) )

    def __len__(self):
        return self.num_sets * self.expansion

    def __getitem__(self, idx_in):
        if idx_in < self.num_sets:
            return self.the_data[idx_in]
        else:
            step = 120 // self.expansion
            roll_index = idx_in // self.num_sets
            assert 1 <= roll_index < self.expansion
            roll_amount = roll_index * step
            assert 1 <= roll_amount < 120

            # get the data
            x, y = self.the_data[idx_in % self.num_sets]

            # Make a copy of the original x tensor
            rolled_x = x.clone().squeeze()

            # Specify the portion of the tensor you want to roll
            slice_indices = (slice(68,188), slice(0, 255))

            # Extract the portion of the tensor to roll
            portion_to_roll = rolled_x[slice_indices]

            # Roll the portion along the first dimension by 1 position
            rolled_portion = torch.roll(portion_to_roll, shifts=roll_amount, dims=0)

            # Replace the rolled portion in the copied tensor
            rolled_x[slice_indices] = rolled_portion

            # return result
            return rolled_x.unsqueeze(0), y

