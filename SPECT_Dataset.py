import torch
from torch.utils.data import Dataset
import numpy as np


def read_float32_binary(file_name):
    # Read binary file as numpy array
    data_np = np.fromfile(file_name, dtype=np.float32)

    # Convert numpy array to PyTorch tensor
    return data_np


class SPECT_Dataset(Dataset):
    def __init__(self, input_prefix, input_suffix, label_prefix, label_suffix, num_sets, normalize_input, normalize_label):
        self.input_prefix = input_prefix
        self.input_suffix = input_suffix
        self.label_prefix = label_prefix
        self.label_suffix = label_suffix
        self.num_sets = num_sets
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

        # reshape input_data from 120 x 250 to 128 to 128
        padded_image = np.pad(input_data.reshape(120, 250), ((4, 4), (3, 3)), mode='constant')
        # reduce to 128 x 128
        input_data = (torch.tensor(np.add.reduceat(padded_image, np.arange(0, padded_image.shape[1], 2), axis=1)).
                      view(1, 128, 128)).to(self.device)

        # reshape label_data from 250 x 250 to 256 x 256 and then compress
        padded_image = np.pad(label_data.reshape(250, 250), ((3, 3), (3, 3)), mode='constant')
        # reduce to 128 x 128
        reduced_data = np.add.reduceat(padded_image, np.arange(0, padded_image.shape[1], 2), axis=1)
        label_data = (torch.tensor(np.add.reduceat(reduced_data, np.arange(0, reduced_data.shape[0], 2), axis=0)).
                      view(1, 128, 128)).to(self.device)

        # normalize as needed
        if self.normalize_input:
            input_data /= torch.max(input_data)

        if self.normalize_label:
            label_data /= torch.max(label_data)

        show_data = False
        if show_data:
            import matplotlib.pyplot as plt
            image = input_data[0]
            plt.figure()
            plt.imshow(image, cmap='gray')  # Assuming grayscale images (adjust colormap if needed)
            plt.show()
            plt.figure()
            image = label_data[0]
            plt.imshow(image, cmap='gray')  # Assuming grayscale images (adjust colormap if needed)
            plt.show()

        return input_data, label_data

