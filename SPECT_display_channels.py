# pull in the trainer module
import sys
import time
import SPECT_Dataset
import SPECT_Dataset_Channels
from PatRecon import trainer
from torch.utils.data import DataLoader, random_split
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import SPECT_Model_channelized
import torch.nn.functional as F


def show_image(left, middle, right):
    # normalize
    normalize_left = True
    normalize_middle = True
    normalize_right = True

    # merge the images
    left_np = left.squeeze()  # Remove the first axis if it's just of size 1
    middle_np = middle.squeeze()
    right_np = right.squeeze()

    # execute normalization
    if normalize_left:
        left_np = left_np / torch.max(left_np)
    if normalize_middle:
        middle_np = middle_np / torch.max(middle_np)
    if normalize_right:
        right_np = right_np / torch.max(right_np)

    # left is [128, 256] while the others are [256, 256]
    left_np = F.pad(left_np, (64, 64, 64, 64))

    # Concatenate along the second axis
    merged_tensor = torch.cat([left_np, middle_np, right_np], axis=1)
    merged_array = merged_tensor.detach().numpy()

    # merged_array is now a numpy 2D array of shape [128, 256]
    plt.figure()
    plt.imshow(merged_array)
    plt.show()


def show_components(components):
    # components are [256x256]. Put them on a 2x4 grid
    components = components.squeeze()
    top = torch.cat([components[0,:,:], components[1,:,:], components[2,:,:], components[3,:,:]], axis=1)
    bottom = torch.cat([components[4,:,:], components[5,:,:], components[6,:,:], components[7,:,:]], axis=1)
    merged_tensor = torch.cat((top, bottom), dim=0)
    merged_array = merged_tensor.detach().numpy()

    # merged_array is now a numpy 2D array of shape [128, 256]
    plt.figure()
    plt.imshow(merged_array)
    plt.show()


def main():
    use_gpu = True if os.environ.get('USER') == 'scott_metzler' and os.environ.get(
        'GROUP') == 'scott_metzler' else False
    if use_gpu:
        print("GPUs will be used.")
    else:
        print("CPU only.")

    # spect_path = '/Users/metzler/Work/Python/SPECT_DL/TestData_26Feb2024_distortions/'
    # num_sets = 25
    spect_path = '/Users/metzler/Work/Python/SPECT_DL/'
    if use_gpu:
        spect_path = '/data/scott_metzler/'
    print(f"Data path is {spect_path}.")
    proj_path = spect_path + "TrainData/AttenProj/"
    phantom_path = spect_path + "TrainData/Phantoms/"
    num_sets = 250

    # load the model
    my_model = SPECT_Model_channelized.SPECT_Model_channelized()
    model_load = False
    while not model_load:
        print("Using path: " + spect_path)
        file_path = []
        file_name = input("Please enter file name (start with '/' to enter full path or 'exit' to abort): ")
        if file_name.lower() == 'exit':
            print("Exiting.")
            sys.exit(0)
        elif file_name[0] == '/':
            file_path = file_name
        else:
            file_path = spect_path + file_name
        try:
            my_model = my_model.load_and_replace(file_path)
            model_load = True
        except Exception as e:
            print(f"Error loading file: {e}.")
            print("Please try again.")
            model_load = False

    # load the data
    dataset = SPECT_Dataset_Channels.SPECT_Dataset_Channels(proj_path, '.atten.noisy.proj',
                                                            phantom_path, '.phantom', num_sets,
                                                            True, True)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # determine if all are being shown or just a special one
    proj_scale = 20
    label_scale = 10
    while True:
        input_str = input(f"Please enter index of data set [0,{num_sets}), 'all', or 'exit': ")
        if input_str.lower() == "exit":
            break
        elif input_str.lower() == "all":
            delay = -1
            break_out = False
            while delay < 0:
                try:
                    delay = float(input(f"Please enter the delay in seconds between images (-1 to cancel): "))
                    if delay < 0:
                        delay = 0
                        break_out = True
                except:
                    print("Unable to parse delay.")
                    break_out = True
            if break_out:
                continue

            # loop
            for index in range(num_sets):
                proj, label = dataset.__getitem__(index)
                try:
                    recon = my_model.forward(proj.view(1, 1, 128, 256))
                    left = proj_scale*proj.squeeze()
                    assert label.shape == (9, 256, 256), "Wrong shape for label"
                    middle = label_scale*SPECT_Dataset_Channels.dechannelize(label.unsqueeze(0))
                    assert recon.shape == (1, 9, 256, 256), "Wrong shape for recon"
                    right = SPECT_Dataset_Channels.dechannelize(recon)
                    show_image(left, middle, right)
                    time.sleep(delay)
                except:
                    print("Error creating or showing image.")
        else:
            try:
                index = int(input_str)
            except:
                print("Unable to parse index.")
                continue
            if 0 <= index < num_sets:
                proj, label = dataset.__getitem__(index)
                try:
                    recon = my_model.forward(proj.view(1, 1, 128, 256))
                    left = proj_scale*proj.squeeze()
                    assert label.shape == (9, 256, 256), "Wrong shape for label"
                    middle = label_scale*SPECT_Dataset_Channels.dechannelize(label.unsqueeze(0))
                    assert recon.shape == (1, 9, 256, 256), "Wrong shape for recon"
                    right = SPECT_Dataset_Channels.dechannelize(recon)
                    show_image(left, middle, right)
                    show_components(recon)
                except:
                    print("Error creating or showing image.")
            else:
                print("The index is outside of the allowed range.")


if __name__ == "__main__":
    # execute main
    main()
