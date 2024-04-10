import torch


def get_tensor_info(image):
    grid_size = image.shape[1]
    assert image.shape[2] == grid_size  # make sure it is square
    assert grid_size != 0 and (grid_size & (grid_size - 1)) == 0, "Grid size is not a power of 2."
    return grid_size, grid_size ** 2


def channelize(image):
    # Convert image to channelized format
    t_size, total_size = get_tensor_info(image)

    # initialize result
    channels = image.shape[0]
    result = torch.zeros(channels, 1, total_size, device=image.device)

    # re-organize
    in2 = image.view(channels, t_size // 2, 2, t_size // 2, 2)

    # get the averages
    avg = torch.mean(in2, dim=(2, 4))

    # extract the element data
    my_ref = result[:, 0, 0:(total_size * 3 // 4)].view(-1, t_size // 2, t_size // 2, 3)
    my_ref[:, :, :, 0] = in2[:, :, 0, :, 0] - avg
    my_ref[:, :, :, 1] = in2[:, :, 0, :, 1] - avg
    my_ref[:, :, :, 2] = in2[:, :, 1, :, 0] - avg

    # fill in the rest
    result[:, 0, (total_size * 3 // 4):total_size].view(-1, t_size // 2, t_size // 2)[:, :, :] = avg if t_size == 2 \
        else channelize(avg)

    # return result
    return result.view(-1, t_size, t_size)


def dechannelize(image):
    # Convert image to channelized format
    t_size, total_size = get_tensor_info(image)

    # initialize result
    channels = image.shape[0]
    result = torch.zeros(channels, t_size // 2, 2, t_size // 2, 2, device=image.device)

    # re-organize
    in2 = image.view(channels, 1, total_size)

    # get the average
    avg = in2[:, 0, 3].view(-1, 1, 1) if t_size == 2 else dechannelize(in2[:, 0, (total_size * 3 // 4):total_size]
                                                                       .view(-1, t_size // 2, t_size // 2))

    # extract the element data
    my_ref = in2[:, 0, 0:(total_size * 3 // 4)].view(-1, t_size // 2, t_size // 2, 3)
    result[:, :, 0, :, 0] = my_ref[:, :, :, 0] + avg
    result[:, :, 0, :, 1] = my_ref[:, :, :, 1] + avg
    result[:, :, 1, :, 0] = my_ref[:, :, :, 2] + avg
    result[:, :, 1, :, 1] = avg - torch.sum(my_ref[:, :, :, 0:3], dim=3)

    # return the result
    return result.view(-1, t_size, t_size)


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


def run_tests():
    # correctness
    for number in range(1, 10):
        print(f"Beginning test of {2 ** number}x{2 ** number}.")
        if mytest_channelized_helper(number):
            print("\tTest passed.")
        else:
            print("\tTest Failed.")

    # timing
    import time
    num_trials = 100
    start_time = time.time()
    OK = True
    for i_trial in range(num_trials):
        for number in range(1, 10):
            if not mytest_channelized_helper(number):
                OK = False

    end_time = time.time()
    if not OK:
        print("\tTest Failed.")
    print(f"Test took {end_time - start_time} seconds.")
    print(f"Test took {(end_time - start_time)/num_trials} seconds per pass.")


# timing
if __name__ == "__main__":
    run_tests()
