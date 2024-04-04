import torch
from torch.utils.data import Dataset
import numpy as np


def read_float32_binary(file_name):
    # Read binary file as numpy array
    data_np = np.fromfile(file_name, dtype=np.float32)

    # Convert numpy array to PyTorch tensor
    return data_np


""""
def channelize(numpy_array):
#This assumes an input of [256,256] and converts is to 9 channels [9, 256, 256].
#The result can be converted back by value(i,j) = \sum_c result(c,i,j)
    assert np.shape(numpy_array) == (256, 256), "Wrong input size in channelize."

    # initialize result
    # Convert the numpy array to a PyTorch tensor
    tensor = torch.tensor(numpy_array).float()
    # Reshape the tensor to have 8 channels
    tensor = tensor.unsqueeze(0).repeat(9, 1, 1)
    assert tensor.shape == (9, 256, 256)

    # calculate result
    for channel in range(9):
        if __debug:
            print(f"Averages for channel {channel}:")

        # Iterate over blocks of 128x128 and calculate the average for each block
        block_size = 256 // int(pow(2, channel))
        num_blocks = 256 // block_size
        for i in range(num_blocks):
            for j in range(num_blocks):
                # Calculate the indices for the current block
                start_i = i * block_size
                end_i = (i + 1) * block_size
                start_j = j * block_size
                end_j = (j + 1) * block_size

                # Extract the current block
                block = tensor[channel, start_i:end_i, start_j:end_j]

                # Calculate the average for each channel in the block
                avg = torch.mean(block, dim=(0, 1))

                # set the average
                tensor[channel, start_i:end_i, start_j:end_j] = avg

                # print
                if __debug:
                    expected = ((2*j+1)*block_size-1)/2 + ((2*i+1)*block_size-1)*256/2
                    if abs(avg-expected)>0.001:
                        print(f"\t({i}{j} = {avg}; Expected = {expected}")

    # copy
    average = tensor.clone()

    # subtract off the averages from the previous layer
    for channel in range(8):
        tensor[channel+1, :, :] -= average[channel, :, :]

    # return result
    return tensor
"""

import torch

def channelize_tensor(tensor):
    """
    This assumes an input tensor with shape [N, 1, 256, 256], where the last
    two dimensions are channelized into 9 channels [N, 9, 256, 256].
    """
    # Ensure input tensor has the correct shape and dimensions
    assert len(tensor.shape) == 4, "Input tensor must be 4-dimensional"
    assert tensor.shape[1] == 1, "The number of input channels must be 1."
    assert tensor.shape[-2:] == (256, 256), "Last two dimensions must have size 256x256"

    # Reshape the tensor to have 9 channels
    tensor_copy = tensor.clone()
    tensor_copy = tensor_copy.repeat(1, 9, 1, 1)
    assert tensor.shape[1] == 1, "Channel dimension of original must have size 9"
    assert tensor_copy.shape[1] == 9, "Channel dimension must have size 9"

    # calculate result
    for channel in range(9):
        # Iterate over blocks of 128x128 and calculate the average for each block
        block_size = 256 // int(pow(2, channel))
        num_blocks = 256 // block_size
        for i in range(num_blocks):
            for j in range(num_blocks):
                # Calculate the indices for the current block
                start_i = i * block_size
                end_i = (i + 1) * block_size
                start_j = j * block_size
                end_j = (j + 1) * block_size

                # Extract the current block
                block = tensor[:, 0, start_i:end_i, start_j:end_j]

                # Calculate the average for each channel in the block
                avg = torch.mean(block, dim=(1, 2))

                # set the average
                tensor_copy[:, channel, start_i:end_i, start_j:end_j] = avg.view(-1, 1, 1)

    # subtract off the averages from the previous layer
    average = tensor_copy.clone()
    for channel in range(8):
        tensor_copy[:, channel+1, :, :] -= average[:, channel, :, :]

    # return result
    return tensor_copy


def dechannelize(tensor):
    """
    Takes a PyTorch tensor with shape [N, 9, 256, 256] and returns a tensor
    with shape [N, 1, 256, 256], where each pixel is the sum over the 9 channels
    of the input tensor.
    """
    # Ensure input tensor has the correct shape and dimensions
    assert len(tensor.shape) == 4, "Input tensor must be 4-dimensional"
    assert tensor.shape[1] == 9, "The number of input channels must be 9."
    assert tensor.shape[-2:] == (256, 256), "Last two dimensions must have size 256x256"

    # Sum the tensor along the channel dimension (dimension 0)
    sum_tensor = torch.sum(tensor, dim=1).unsqueeze(1)
    assert len(sum_tensor.shape) == 4, "Output tensor must be 4-dimensional"
    assert sum_tensor.shape[1] == 1, "The number of output channels must be 1."
    assert sum_tensor.shape[-2:] == (256, 256), "Last two dimensions must have size 256x256."
    assert tensor.shape[0] == sum_tensor.shape[0], "Number of output instances should match that of input."

    return sum_tensor


class SPECT_Dataset_Channels(Dataset):
    """
    This class will take the sinogram input and pad it as in SPECT_Dataset. However, the label
    image will be converted to channels with different resolutions. The first will be the full-resolution
    image, but will be average subtracted.
    """

    def __init__(self, input_prefix, input_suffix, label_prefix, label_suffix, num_sets, normalize_input, normalize_label):
        self.input_prefix = input_prefix
        self.input_suffix = input_suffix
        self.label_prefix = label_prefix
        self.label_suffix = label_suffix
        self.num_sets = num_sets
        self.normalize_input = normalize_input
        self.normalize_label = normalize_label

    def __len__(self):
        return self.num_sets

    def __getitem__(self, idx):
        # Define a dictionary to cache computed results
        if not hasattr(self, 'cache'):
            self.cache = {}

        # Check if the result for the given index is already cached
        if idx in self.cache:
             return self.cache[idx]

        # Calculate the result for the given index
        result = self.__getitem__impl(idx)

        # Cache the result for future use
        self.cache[idx] = result

        return result

    def __getitem__impl(self, idx):
        # Load input and label binary files for the given index
        in_name = self.input_prefix + str(idx) + self.input_suffix
        label_name = self.label_prefix + str(idx) + self.label_suffix
        input_data = read_float32_binary(in_name)
        label_data = read_float32_binary(label_name)

        # reshape input_data from 120 x 250 to 128 to 256
        padded_input = np.pad(input_data.reshape(120, 250), ((4, 4), (3, 3)), mode='constant')

        # reshape label_data from 250 x 250 to 256 x 256 and then channelize
        padded_label = np.pad(label_data.reshape(250, 250), ((3, 3), (3, 3)), mode='constant')
        channelized_label = channelize_tensor(torch.tensor(padded_label).unsqueeze(0).unsqueeze(0)).squeeze()

        # normalize as needed
        if self.normalize_input:
            padded_input /= np.max(padded_input)

        if self.normalize_label:
            channelized_label /= torch.max(channelized_label)

        show_data = False
        if show_data:
            # fluff the channels
            recreated = self.dechannelize(channelized_label)
            import matplotlib.pyplot as plt
            image = input_data[0]
            plt.figure()
            plt.imshow(image, cmap='gray')  # Assuming grayscale images (adjust colormap if needed)
            plt.show()
            plt.figure()
            image = label_data[0]
            plt.imshow(image, cmap='gray')  # Assuming grayscale images (adjust colormap if needed)
            plt.show()

        return torch.tensor(padded_input).unsqueeze(0), channelized_label


__debug = False
