import torch
from torch.utils.data import Dataset
import numpy as np
#import time

"""
import threading

# global timing for this file
total_time = 0

# Create a lock object
lock = threading.Lock()

def add_to_time(value):
    global total_time
    # Acquire the lock
    lock.acquire()
    try:
        # Update the variable
        total_time += value
    finally:
        # Release the lock
        lock.release()

def get_time():
    return total_time
"""

def read_float32_binary(file_name):
    # Read binary file as numpy array
    data_np = np.fromfile(file_name, dtype=np.float32)

    # Convert numpy array to PyTorch tensor
    return data_np


class SPECT_Dataset3(Dataset):
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

    def __getitem__(self, idx_in):
        #start_time = time.time()
        idx = idx_in % self.num_sets

        # Load input and label binary files for the given index
        in_name = self.input_prefix + str(idx) + self.input_suffix
        label_name = self.label_prefix + str(idx) + self.label_suffix
        input_data = read_float32_binary(in_name)
        label_data = read_float32_binary(label_name)

        # reshape input_data from 120 x 250 to 128 to 128
        if idx_in < self.num_sets:
            padded_image = np.pad(input_data.reshape(120, 250), ((68, 68), (3, 3)), mode='constant')
            input_data = torch.tensor(padded_image, device=self.device).view(1, 256, 256)
        else:
            step = 120 // self.expansion
            roll_index = idx_in // self.num_sets
            assert 1 <= roll_index < self.expansion
            roll_amount = roll_index * step
            assert 1 <= roll_amount < 120
            padded_image = np.pad(np.roll(input_data.reshape(120, 250), roll_amoumt, axis=0), ((68, 68), (3, 3)), mode='constant')
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

        #this_time = time.time() - start_time
        #add_to_time(this_time)
        #g_time = get_time()
        return input_data, label_data

