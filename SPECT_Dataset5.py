import torch
from torch.utils.data import Dataset
import numpy as np

def read_float32_binary(file_name):
    # Read binary file as numpy array
    data_np = np.fromfile(file_name, dtype=np.float32)

    # Convert numpy array to PyTorch tensor
    return data_np

class SPECT_Dataset5(Dataset):
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

    def __len__(self):
        return self.num_sets

    def __getitem__(self, idx):
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

        return input_data, label_data

