import torch


def set_global_scale(tensor, value):
    tensor[:, 0, 0] = value

def set_scale_factors(tensor, this_n, this_total_size, level, values):
    assert 1 <= level < this_n
    indices = get_indices(this_n, level)
    set_values(tensor, this_total_size, indices, values)

def set_grid_parameters(tensor, this_n, this_total_size, values):
    indices = get_indices(this_n, this_n - 1)
    indices2 = indices[1], this_total_size
    set_values(tensor, this_total_size, indices2, values)

def get_global_scale(tensor):
    return tensor[:, 0, 0]

def get_scale_factors(tensor, this_n, this_total_size, level):
    assert level < this_n
    indices = get_indices(this_n, level)
    return get_values(tensor, this_total_size, indices)

def get_grid_parameters(tensor, this_n, this_total_size):
    indices = get_indices(this_n, this_n - 1)
    indices2 = indices[1], this_total_size
    return get_values(tensor, this_total_size, indices2)

def get_tensor_info(image):
    grid_size = image.shape[1]
    assert image.shape[2] == grid_size  # make sure it is square
    assert is_power_of_two(grid_size)
    this_n = find_power_of_two(grid_size)
    this_tensor_size = grid_size
    this_total_size = this_tensor_size ** 2
    return grid_size, this_n, this_tensor_size, this_total_size

def channelize(image):
    # Convert image to channelized format
    grid_size, this_n, this_tensor_size, this_total_size = get_tensor_info(image)

    # initialize result
    channels = image.shape[0]
    result = torch.zeros(channels, this_tensor_size, this_tensor_size, device=image.device)

    # check the size
    assert image.shape == result.shape

    # Get the global average
    last_averages = torch.mean(image, dim=(1, 2))
    set_global_scale(result, last_averages)
    last_averages = last_averages.unsqueeze(1)

    # Get the scale factors by level
    for level in range(1, this_n+1):
        # Iterate over blocks of 128x128 and calculate the average for each block
        block_size = this_tensor_size // int(pow(2, level))
        num_blocks = this_tensor_size // block_size

        # allocate an array
        num_values = 3 * 2 ** (2 * (level - 1))
        the_values = torch.zeros(channels, num_values, device=image.device)
        num_values2 = 4 * 2 ** (2 * (level - 1))
        next_averages = torch.zeros(channels, num_values2, device=image.device)
        counter = 0
        counter2 = 0

        for i in range(num_blocks):
            for j in range(num_blocks):
                # Calculate the indices for the current block
                start_i = i * block_size
                end_i = (i + 1) * block_size
                start_j = j * block_size
                end_j = (j + 1) * block_size

                # Extract the current block
                block = image[:, start_i:end_i, start_j:end_j]

                # Calculate the average for each channel in the block
                # Store all the averages for next round, but skip tensor if otherwise calculated
                the_mean = torch.mean(block, dim=(1, 2))
                if not skip_block(i, j):  # skips last of every 2x2
                    values_index = (i // 2) * (num_blocks // 2) + (j // 2)
                    assert values_index < num_values, (f"Index error for level {level}, (i,j)=({i},{j}), with "
                                                       f"{num_blocks} blocks of size {block_size}.")
                    the_values[:, counter] = the_mean - last_averages[:, values_index]
                    counter += 1
                next_averages[:, counter2] = the_mean
                counter2 += 1

        # check the count
        assert counter == num_values
        assert counter2 == num_values2

        # store
        if level == this_n:
            set_grid_parameters(result, this_n, this_total_size, the_values)
        else:
            set_scale_factors(result, this_n, this_total_size, level, the_values)

        # set last
        last_averages = next_averages

    # return the result
    return result

def dechannelize(image):
    # Convert channelized format back to regular format
    grid_size, this_n, this_tensor_size, this_total_size = get_tensor_info(image)

    # initialize image
    # initialize result
    channels = image.shape[0]
    result = torch.zeros(channels, this_tensor_size, this_tensor_size, device=image.device)
    max_num_values = 3 * 2 ** (2 * (this_n - 1))
    the_sum = torch.zeros(channels, max_num_values, device=image.device)

    # check the size
    assert image.shape == result.shape

    # Get the global average
    result[:, :, :] = get_global_scale(image).view(-1, 1, 1)

    # Get the scale factors by level
    for level in range(1, this_n + 1):
        # Iterate over blocks of 128x128 and calculate the average for each block
        block_size = this_tensor_size // int(pow(2, level))
        num_blocks = this_tensor_size // block_size

        # allocate an array
        num_values = 3 * 2 ** (2 * (level - 1))
        counter = 0

        # reset the sum tensor
        if level != 1:
            the_sum.fill_(0)

        # get the scale factors for this level
        if level == this_n:
            the_values = get_grid_parameters(image, this_n, this_total_size)
        else:
            the_values = get_scale_factors(image, this_n, this_total_size, level)

        """
        for i in range(num_blocks):
            for j in range(num_blocks):
                # Calculate the indices for the current block
                start_i = i * block_size
                end_i = (i + 1) * block_size
                start_j = j * block_size
                end_j = (j + 1) * block_size

                # See if this block is skipped
                values_index = (i // 2) * (num_blocks // 2) + (j // 2)
                if not skip_block(i, j):  # skips last of every 2x2
                    result[:, start_i:end_i, start_j:end_j] += the_values[:, counter].view(-1, 1, 1)
                    the_sum[:, values_index] += the_values[:, counter]
                    counter += 1
                else:
                    result[:, start_i:end_i, start_j:end_j] -= the_sum[:, values_index].view(-1, 1, 1)
        """

        # Lists to store i, j combinations that are skipped and not skipped
        skip_indices = []
        non_skip_indices = []

        for i in range(num_blocks):
            for j in range(num_blocks):
                if skip_block(i, j):
                    skip_indices.append((i, j))
                else:
                    non_skip_indices.append((i, j))

        for i, j in non_skip_indices:
            # Calculate the indices for the current block
            start_i = i * block_size
            end_i = (i + 1) * block_size
            start_j = j * block_size
            end_j = (j + 1) * block_size

            values_index = (i // 2) * (num_blocks // 2) + (j // 2)
            result[:, start_i:end_i, start_j:end_j] += the_values[:, counter].view(-1, 1, 1)
            the_sum[:, values_index] += the_values[:, counter]
            counter += 1

        for i, j in skip_indices:
            # Calculate the indices for the current block
            start_i = i * block_size
            end_i = (i + 1) * block_size
            start_j = j * block_size
            end_j = (j + 1) * block_size

            values_index = (i // 2) * (num_blocks // 2) + (j // 2)
            result[:, start_i:end_i, start_j:end_j] -= the_sum[:, values_index].view(-1, 1, 1)

        # check the count
        assert counter == num_values

    # return the result
    return result

def get_indices(this_n, level):
    assert level < this_n
    start_index = 1 + 3 * sum([2 ** (2 * (l - 1)) for l in range(1, level)])
    end_index = 1 + 3 * sum([2 ** (2 * (l - 1)) for l in range(1, level + 1)])
    return start_index, end_index

def set_values(tensor, this_total_size, indices, values):
    length = indices[1] - indices[0]
    assert values.shape[1] == length
    tensor.view(-1, 1, this_total_size)[:, 0, indices[0]:indices[1]] = values

def get_values(tensor, this_total_size, indices):
    return tensor.view(-1, 1, this_total_size)[:, 0, indices[0]:indices[1]]

def skip_block(i, j):
    return i % 2 == 1 and j % 2 == 1


def is_power_of_two(n):
    return n != 0 and (n & (n - 1)) == 0


def find_power_of_two(two_to_the_n):
    n = 0
    while two_to_the_n != 1:
        two_to_the_n >>= 1
        n += 1

    return n


def mytest_channelized_helper(n):
    result = True

    test_dim = 2 ** n
    values = torch.linspace(0., test_dim * test_dim - 1, steps=test_dim * test_dim)
    tensor = values.reshape(test_dim, test_dim)
    multiplied_tensors = []

    # Multiply the original tensor by (i+1) and store the results in the list
    num_channels = 10
    for i in range(num_channels):
        multiplied_tensor = tensor * (i + 1)
        multiplied_tensors.append(multiplied_tensor.unsqueeze(0))
    test_image = torch.cat(multiplied_tensors, dim=0).to(torch.float64)  # [num_channels, 256, 256]

    # channelize and de-channelize
    channelized_image = channelize(test_image)
    dechannelized_image = dechannelize(channelized_image)

    # test
    if not torch.all(torch.abs(dechannelized_image - test_image) < 0.01):
        print(f"De-channelized values for {test_dim}x{test_dim} did not match.")
        result = False
        for i in range(test_dim):
            for j in range(test_dim):
                if not torch.all(torch.abs(dechannelized_image[:, i, j] - test_image[:, i, j]) < 0.01):
                    print(f"Element [:, {i}, {j}] does not match: {dechannelized_image[:, i, j]}; "
                          f"{test_image[:, i, j]}")

    return result


def mytest_channelized():
    # Loop over the list
    #for number in range(1, 10):
    for number in range(1, 10):
        print(f"Beginning test of {2 ** number}x{2 ** number}.")
        if mytest_channelized_helper(number):
            print("\tTest passed.")
        else:
            print("\tTest Failed.")


#if __name__ == "__main__":
#    mytest_channelized()
