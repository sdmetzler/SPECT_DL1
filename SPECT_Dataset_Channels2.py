import torch
from torch.utils.data import Dataset
import numpy as np


def read_float32_binary(file_name):
    # Read binary file as numpy array
    data_np = np.fromfile(file_name, dtype=np.float32)

    # Convert numpy array to PyTorch tensor
    return data_np


def channelize_tensor(tensor, save_scale=True, higher_avg=None):  # higher_avg is the sum of all averages above this point
    """
    This assumes an input tensor with shape [N, 1, 2**i, 2**i]. This will be channelized
    into an image of the same size in a fractal way, saving the average of each group, except for the last.
    """
    # Ensure input tensor has the correct shape and dimensions
    assert len(tensor.shape) == 4, "Input tensor must be 4-dimensional"
    assert tensor.shape[1] == 1, "The number of input channels must be 1."
    num_elems = tensor.shape[2]
    assert tensor.shape[-2:] == (num_elems, num_elems), "Last two dimensions must be equal."

    # setup higher_average
    num_entries = tensor.shape[0]
    if higher_avg is None:
        higher_avg = torch.zeros(num_entries)


    # get the recursive result for smaller, unless the size here is 2x2
    tensor_result = []
    if num_elems == 2:
        tensor_result = torch.zeros(num_entries, 1, 1, 4 if save_scale else 3)
        avg = torch.mean(tensor, dim=(2, 3)).view(-1)  # files num_entries elements
        offset = 0
        if save_scale:
            tensor_result[:, 0, 0, 0] = avg
            offset = 1
        tensor_result[:, 0, 0, offset] = tensor[:, 0, 0, 0] - avg
        tensor_result[:, 0, 0, offset+1] = tensor[:, 0, 0, 1] - avg
        tensor_result[:, 0, 0, offset+2] = tensor[:, 0, 1, 0] - avg
        if save_scale:
            tensor_result = tensor_result.view(-1, 1, 2, 2)
    else:
        new_elems = num_elems // 2
        t1 = tensor[:, :, 0:new_elems,         0:new_elems]
        t2 = tensor[:, :, 0:new_elems,         new_elems:num_elems]
        t3 = tensor[:, :, new_elems:num_elems, 0:new_elems]
        t4 = tensor[:, :, new_elems:num_elems, new_elems:num_elems]
        avg = torch.mean(tensor, dim=(2, 3)).view(-1)  # files num_entries elements
        t1_offset = torch.mean(t1, dim=(2, 3)).view(-1) - avg - higher_avg
        t2_offset = torch.mean(t2, dim=(2, 3)).view(-1) - avg - higher_avg
        t3_offset = torch.mean(t3, dim=(2, 3)).view(-1) - avg - higher_avg
        t4_offset = torch.mean(t4, dim=(2, 3)).view(-1) - avg - higher_avg
        r1 = channelize_tensor(t1, True, t1_offset)
        r2 = channelize_tensor(t2, True, t2_offset)
        r3 = channelize_tensor(t3, True, t3_offset)
        r4 = channelize_tensor(t4, False, t4_offset)

        # put it all in the result
        r1_length = r1.shape[2] * r1.shape[3]
        r2_length = r2.shape[2] * r2.shape[3]
        r3_length = r3.shape[2] * r3.shape[3]
        r4_length = r4.shape[2] * r4.shape[3]
        r_length = 1 + r1_length + r2_length + r3_length + r4_length
        tensor_result = torch.zeros(num_entries, r_length)  # make it a stream until the end

        # save it
        tensor_result[:, 0] = avg
        store_count = int(1)

        # r1
        tensor_result[:, store_count:(store_count+r1_length)] = r1.view(-1, r1_length)
        store_count += r1_length

        # r2
        tensor_result[:, store_count:(store_count+r2_length)] = r2.view(-1, r2_length)
        store_count += r2_length

        # r3
        tensor_result[:, store_count:(store_count+r3_length)] = r3.view(-1, r3_length)
        store_count += r3_length

        # r4
        tensor_result[:, store_count:(store_count+r4_length)] = r4.view(-1, r4_length)
        store_count += r4_length
        assert store_count == r_length

        # reshape the result
        tensor_result = tensor_result.view(-1, 1, num_elems, num_elems)

    # return
    return tensor_result


def dechannelize_tensor(tensor):
    """
    Takes a PyTorch tensor with shape [N, 9, num_entries, num_entries] and returns a tensor
    with shape [N, 1, num_entries, num_entries].
    """
    # Ensure input tensor has the correct shape and dimensions
    assert len(tensor.shape) == 4, "Input tensor must be 4-dimensional"
    assert tensor.shape[1] == 1, "The number of input channels must be 1."
    num_elems1 = tensor.shape[2]
    num_elems2 = tensor.shape[3]
    assert num_elems1 == num_elems2 or (num_elems1 == 1 and num_elems2 == 3), "Expected dimensions for input"

    # initialize result
    num_entries = tensor.shape[0]
    tensor_result = None

    # get the recursive result for smaller, unless the size here is 2x2
    if num_elems1 == 1 and num_elems2 == 3:
        tensor_result = torch.zeros(num_entries, 1, 2, 2)
        tensor_result[:, 0, 0, 0] = tensor[:, 0, 0, 0]
        tensor_result[:, 0, 0, 1] = tensor[:, 0, 0, 1]
        tensor_result[:, 0, 1, 0] = tensor[:, 0, 0, 2]
        tensor_result[:, 0, 1, 1] = -tensor[:, 0, 0, 0] - tensor[:, 0, 0, 1] - tensor[:, 0, 0, 2]
    else:
        num_elems = num_elems1
        assert num_elems2 == num_elems
        tensor_result = torch.zeros(num_entries, 1, num_elems, num_elems)
        if num_elems == 2:
            tensor_local = tensor.view(-1, 1, 1, 4)
            avg = tensor_local[:, 0, 0, 0]
            tensor_result[:, 0, 0, 0] = avg + tensor_local[:, 0, 0, 1]
            tensor_result[:, 0, 0, 1] = avg + tensor_local[:, 0, 0, 2]
            tensor_result[:, 0, 1, 0] = avg + tensor_local[:, 0, 0, 3]
            tensor_result[:, 0, 1, 1] = avg - tensor_local[:, 0, 0, 1] - tensor_local[:, 0, 0, 2] - tensor_local[:, 0, 0, 3]
        else:
            new_elems = num_elems // 2

            # read and apply the scale factor
            this_scale = tensor[:, 0, 0, 0]
            count = int(1)

            # Define a lambda function to extract scale and tensor
            extract_scale_and_tensor = lambda c: (
                tensor.view(-1, 1, 1, num_elems**2)[:, 0, 0, c],
                tensor.view(-1, 1, 1, num_elems**2)[:, :, c + 1:(c + 1 + new_elems ** 2)].view(-1, 1, new_elems, new_elems),
                c + 1 + new_elems ** 2
            )
            extract_tensor = lambda c: (
                tensor[:, :, c + 1:(c + 1 + new_elems ** 2)].view(-1, 1, new_elems, new_elems),
                c + new_elems ** 2
            )

            # sub-tensors
            scale1, t1, count = extract_scale_and_tensor(count)
            scale2, t2, count = extract_scale_and_tensor(count)
            scale3, t3, count = extract_scale_and_tensor(count)
            t4, count = extract_tensor(count)
            scale4 = -scale1 - scale2 - scale3

            # dechannelize those
            dt1 = dechannelize_tensor(t1) + scale1 + this_scale
            dt2 = dechannelize_tensor(t2) + scale2 + this_scale
            dt3 = dechannelize_tensor(t3) + scale3 + this_scale
            dt4 = dechannelize_tensor(t4) + scale4 + this_scale

            # set the elements
            tensor_result[:, 0, 0:new_elems        , 0:new_elems]         = dt1
            tensor_result[:, 0, 0:new_elems        , new_elems:num_elems] = dt2
            tensor_result[:, 0, new_elems:num_elems, 0:new_elems]         = dt3
            tensor_result[:, 0, new_elems:num_elems, new_elems:num_elems] = dt4

    # return
    return tensor_result


    assert len(tensor.shape) == 4, "Input tensor must be 4-dimensional"
    assert tensor.shape[1] == 1, "The number of input channels must be 9."
    num_entries = tensor.shape[0]
    assert tensor.shape[-2:] == (num_entries, num_entries), "Last two dimensions must have equal size."

    # Sum the tensor along the channel dimension (dimension 0)
    sum_tensor = torch.sum(tensor, dim=1).unsqueeze(1)
    assert len(sum_tensor.shape) == 4, "Output tensor must be 4-dimensional"
    assert sum_tensor.shape[1] == 1, "The number of output channels must be 1."
    assert sum_tensor.shape[-2:] == (256, 256), "Last two dimensions must have size 256x256."
    assert tensor.shape[0] == sum_tensor.shape[0], "Number of output instances should match that of input."

    return sum_tensor



class SPECT_Dataset_Channels2(Dataset):
    """
    This class will take the sinogram input and pad it as in SPECT_Dataset. However, the label
    image will be converted to channels with different resolutions. The first will be the full-resolution
    image, but will be average subtracted.
    """

    def __init__(self, input_prefix, input_suffix, label_prefix, label_suffix, num_sets, normalize_input,
                 normalize_label):
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
