# pull in the trainer module
import time

import SPECT_Dataset
from PatRecon import trainer
import torch
from torch.utils.data import DataLoader, random_split
import os


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
    num_epochs = 100

    args = Args(spect_path, use_gpu)
    my_trainer = trainer.Trainer_ReconNet(args)
    my_trainer.load()  # resume from previous best

    # start a new wandb run to track this script
    # wandb.login(key='21c49efb26ea0bdb36ad0bd2a05f99d493072bf5')
    if not use_gpu:
        import wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="my-awesome-project",

            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": "CNN",
                "dataset": "SPECT_TRAIN",
                "epochs": num_epochs,
            }
        )

    # set a random seed for reproducibility
    torch.manual_seed(7)

    # load the data
    # Create custom dataset instance
    dataset = SPECT_Dataset.SPECT_Dataset(proj_path, '.atten.noisy.proj',
                                          phantom_path, '.phantom', num_sets,
                                          normalize_input=True, normalize_label=True)

    # Create data loader
    batch_size = 25
    train_size = int(0.8 * num_sets)
    test_size = num_sets - train_size
    train_dataset, validate_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    # record the current time
    start_time = time.time()
    iter_start_time = start_time

    # loop over epochs to train
    losses = []
    for ie in range(num_epochs):
        # train
        train_result = my_trainer.train_epoch(train_loader, ie)

        # log metrics
        losses.append(train_result)
        if not use_gpu:
            this_time = time.time()
            wandb.log({"iter_time": this_time-iter_start_time, "loss": train_result})
            iter_start_time = this_time

        # save the state
        my_trainer.save(train_result, ie)

    # close
    if not use_gpu:
        wandb.finish()

    # plot it
    if not use_gpu:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(num_epochs), losses)
        plt.ylabel("loss/error")
        plt.xlabel("Epoch")
        plt.show()

    # time to train
    train_time = time.time()
    print(f"Total training time: {train_time-start_time} seconds.")

    # validate
    print("Validating")
    my_trainer.validate(validate_loader)
    validate_time = time.time()
    print(f"Total validation time: {validate_time-train_time} seconds.")

if __name__ == "__main__":
    # Call the main function
    print("This program will train the network and save it.")

    # print the environment
    #import sys
    #import os
    #print(sys.executable)
    #print(os.getcwd())
    #print(sys.path)

    # execute main
    main()

    # Done
    print("Done")
