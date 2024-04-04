import torch
from torch.utils.data import Dataset
import numpy as np
import multiprocessing as mp


def read_float32_binary(file_name):
    # Read binary file as numpy array
    data_np = np.fromfile(file_name, dtype=np.float32)

    # Convert numpy array to PyTorch tensor
    return data_np


class SPECT_Dataset2(Dataset):
    def __init__(self, input_prefix, input_suffix, label_prefix, label_suffix, num_sets, normalize_input, normalize_label):
        self.input_prefix = input_prefix
        self.input_suffix = input_suffix
        self.label_prefix = label_prefix
        self.label_suffix = label_suffix
        self.num_sets = num_sets
        self.normalize_input = normalize_input
        self.normalize_label = normalize_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = {}  # Cache to store loaded data
        self.cache_lock = mp.Lock()  # Lock for thread safety

    def __len__(self):
        return self.num_sets

    def __getitem__(self, idx):
        if idx in self.cache:
            # If data for this index is already cached, return it
            input_data, label_data = self.cache[idx]
        else:
            # Load input and label binary files for the given index
            in_name = self.input_prefix + str(idx) + self.input_suffix
            label_name = self.label_prefix + str(idx) + self.label_suffix
            input_data = read_float32_binary(in_name)
            label_data = read_float32_binary(label_name)

            # reshape input_data from 120 x 250 to 128 to 128
            padded_image = np.pad(input_data.reshape(120, 250), ((68, 68), (3, 3)), mode='constant')
            input_data = torch.tensor(padded_image, device=self.device).view(1, 256, 256)

            # reshape label_data from 250 x 250 to 256 x 256 and then compress
            padded_image = np.pad(label_data.reshape(250, 250), ((3, 3), (3, 3)), mode='constant')
            label_data = torch.tensor(padded_image, device=self.device).view(1, 256, 256)

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

            # Cache the loaded data
            #with self.cache_lock:
            #    self.cache[idx] = (input_data, label_data)

        return input_data, label_data

