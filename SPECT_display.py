# pull in the trainer module
import time
import SPECT_Dataset
from PatRecon import trainer
from torch.utils.data import DataLoader, random_split
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch


# create the trainer
class Args:
    def __init__(self, the_path, gpu=False):
        self.exp = the_path + 'exp/'
        self.arch = 'ReconNet'
        self.print_freq = 1
        self.output_path = the_path + 'out/'
        self.resume = 'best'  # args.resume
        self.num_views = 1  # treat the entire sinogram as a single view
        self.output_channel = 1  # size of output
        self.init_gain = 0.01  # I think this is the learning rate
        self.init_type = 'init_type'
        self.cuda = gpu  # I added some if blocks to work with MacOS
        self.loss = 'l2'
        self.optim = 'adam'
        self.lr = 0.01
        self.weight_decay = 0.01


def show_image(left, middle, right):

    # merge the images
    left_np = left.squeeze()  # Remove the first axis if it's just of size 1
    middle_np = middle.squeeze()
    right_np = right.squeeze()

    # Concatenate along the second axis
    merged_tensor = torch.cat([left_np, middle_np, right_np], axis=1)
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

    args = Args(spect_path, use_gpu)
    my_trainer = trainer.Trainer_ReconNet(args)

    # load
    my_trainer.load()

    # load the data
    dataset = SPECT_Dataset.SPECT_Dataset(proj_path, '.atten.noisy.proj',
                                          phantom_path, '.phantom', num_sets, True, True)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # determine if all are being shown or just a special one
    proj_scale = 20
    label_scale = 10
    while True:
        input_str = input(f"Please enter index of data set [0,{num_sets}), 'all', or 'exit': ")
        if input_str == "exit":
            break
        elif input_str == "all":
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
                    recon = my_trainer.model.forward(proj.view(1, 1, 128, 128))
                    show_image(proj*proj_scale, label*label_scale, recon)
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
                    recon = my_trainer.model.forward(proj.view(1,1,128,128))
                    show_image(proj*proj_scale, label*label_scale, recon)
                except:
                    print("Error creating or showing image.")
            else:
                print("The index is outside of the allowed range.")


if __name__ == "__main__":
    # execute main
    main()
