# pull in the trainer module
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import os
import SPECT_Dataset4
import CustomDataLoader
import SPECT_Model_channelized2
import numpy as np
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
import multiprocessing
import gc
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main(args):
    use_gpu = torch.cuda.is_available()
    if args.verbose:
        if use_gpu:
            print("GPUs will be used.")
        else:
            print("CPU only.")
 
    # set multi-processing
    if use_gpu:
        mp.set_start_method('spawn')  # Set start method to 'spawn' for CUDA compatibility

    # get the device
    device = torch.device("cuda" if use_gpu else "cpu")

    # start recording memory snapshots
    if args.memory_management and use_gpu:
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    # Prefix for file names.
    TIME_FORMAT_STR = "%Y-%m-%d_%H:%M:%S"
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    memory_file_prefix = f"memory_snapshot_{args.channelize}_{args.go_to_2x2}_{args.go_to_1x1}_{timestamp}"

    # spect_path
    spect_path_laptop = '/Users/metzler/Work/Python/SPECT_DL/'
    #spect_path_access = '/home/metzler/NN_Test1/SPECT_DL/'
    spect_path_access = '/projects/bcps/smetzler/NN_Trial1/SPECT_DL/'
    spect_path = spect_path_access if use_gpu else spect_path_laptop
    proj_path = spect_path + "TrainData/AttenProj/"
    phantom_path = spect_path + "TrainData/Phantoms/"
    print(f"Data path is {spect_path}.")

    # settings
    num_sets = 250

    # print some more verbose information
    output_path = spect_path + args.output
    if args.verbose:
        print(f"Output will be to {output_path}.")
        print(f"The learning rate is {args.learning_rate}.")

    # start a new wandb run to track this script
    if args.use_wandb:
        import wandb
        wandb.login(key='21c49efb26ea0bdb36ad0bd2a05f99d493072bf5')
        wandb.init(
            # set the wandb project where this run will be logged
            project="SPECT-channelized",

            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": "CNN",
                "dataset": "SPECT_TRAIN",
                "epochs": args.num_epochs,
            }
        )

    # set a random seed for reproducibility
    torch.manual_seed(args.seed)

    # load the data
    # Create custom dataset instance
    start_d_time = time.time()
    dataset = SPECT_Dataset4.SPECT_Dataset4(proj_path, '.atten.noiseless.proj',
                                            phantom_path, '.phantom', num_sets, 10, 
                                            normalize_input=False, normalize_label=False,
                                            add_noise=False)
    print(f"Dataset creating time: {time.time()-start_d_time} sec.")

    # put aside data for testing
    train_batch_size = args.batch_size
    validate_batch_size = args.batch_size
    """
    development_size = int(0.8 * num_sets * args.expansion)
    test_size = num_sets * args.expansion - development_size
    train_size = (development_size * 4) // 5
    validate_size = development_size - train_size
    development_dataset, testing_dataset = random_split(dataset, [development_size, test_size])
    train_dataset, validate_dataset = random_split(development_dataset, [train_size, validate_size])
    """
    development_size = int(num_sets*10)
    train_size = (development_size * 4) // 5
    validate_size = development_size - train_size
    #development_dataset, testing_dataset = random_split(dataset, [development_size, test_size])
    train_dataset, validate_dataset = random_split(dataset, [train_size, validate_size])

    # Create data loader
    num_workers = 2  #multiprocessing.cpu_count()
    if args.verbose:
        print(f"Creating data loaders with {num_workers} workers.")
    #train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    #validate_loader = DataLoader(validate_dataset, batch_size=validate_batch_size, shuffle=True, num_workers=num_workers)
    train_loader = CustomDataLoader.CustomDataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    validate_loader = CustomDataLoader.CustomDataLoader(dataset=validate_dataset, batch_size=validate_batch_size, shuffle=True)

    # create the model
    my_model = SPECT_Model_channelized2.SPECT_Model_channelized2(args.channelize, args.go_to_2x2, args.go_to_1x1)
    if args.reload is not None:
        print(f"Reloading model from: {spect_path + args.reload}")
        my_model = my_model.load_and_replace(spect_path + args.reload)
    my_model = my_model.to(device)
    if args.verbose:
        print(f"Model has {count_parameters(my_model)} parameters.")

    # create the criterion
    criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(my_model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    # scaler allows mixed precision
    #scaler = GradScaler() if use_gpu else None

    # record the current time
    start_time = time.time()
    iter_start_time = start_time

    # record the memory snapshot
    if args.memory_management and use_gpu:
        try:
            torch.cuda.memory._dump_snapshot(f"{memory_file_prefix}.pickle")
        except Exception as e:
            logger.error(f"Failed to capture memory snapshot {e}")

    # loop over epochs to train
    best_validation = 1_000_000.
    for ie in range(args.num_epochs):
        # print info
        print(f"Beginning epoch {ie}.")

        # Train
        time_f = 0
        time_g = 0
        train_loss = 0
        start_e_time = time.time()
        for batch, (X_train, y_train) in enumerate(train_loader):
            # print
            start_f_time = time.time()
            with autocast():  # Enables automatic mixed precision
                y_pred = my_model.forward(X_train)
                loss = criterion(y_pred, y_train)
                assert not torch.isnan(loss), "Training loss is NaN. Exiting."
                train_loss += loss
            end_f_time = time.time()

            # print summary
            if ie==0:
                print(f"Training loss for batch {batch} of epoch {ie} is {loss}.")

            # update our parameters
            #start_g_time = time.time()
            optimizer.zero_grad()
            if use_gpu:
                # update our parameters
                loss.backward()
                optimizer.step()
                #scaler.scale(loss).backward()
                #scaler.step(optimizer)
                #scaler.update()
                torch.cuda.empty_cache()
            else:
                # update our parameters
                loss.backward()
                optimizer.step()

            end_g_time = time.time()

            # record time
            time_f += end_f_time - start_f_time
            time_g += end_g_time - end_f_time  # start_g_time
        train_loss /= (batch + 1)
        print(f"Avg. training loss is {train_loss}.")
        end_e_time = time.time()
        epoch_time = end_e_time - start_e_time
        print(f"Total train time: {epoch_time} sec.")
        print(f"\tTotal forward time: {time_f} sec.")
        print(f"\tTotal gradient time: {time_g} sec.")
        print(f"\tTotal other time: {epoch_time - time_f - time_g} sec.")
        #print(f"\tCummulative get_time: {SPECT_Dataset3.get_time()} sec.")

        # Validate
        validate_loss = 0.
        start_v_time = time.time()
        with torch.no_grad():
            for batch, (X_validate, y_validate) in enumerate(validate_loader):
                y_pred = my_model(X_validate)
                loss = criterion(y_pred, y_validate)
                assert not torch.isnan(loss), "Validation loss is NaN. Exiting."
                validate_loss += loss

                if ie+1 == args.num_epochs and batch == 0 and args.write_tar:
                    fname_x = spect_path + args.write_tar + '_x.tar'
                    fname_y = spect_path + args.write_tar + '_y.tar'
                    fname_p = spect_path + args.write_tar + '_p.tar'
                    if args.verbose:
                        print(f"Writing validation to files {fname_x}, {fname_y}, and {fname_p}.")
                        print(f"Shapes are {X_validate.shape}, {y_validate.shape}, and {y_pred.shape}")
                    torch.save(X_validate, fname_x)
                    torch.save(y_validate, fname_y)
                    torch.save(y_pred, fname_p)

            # get averages
            validate_loss /= (batch + 1)
        end_v_time = time.time()
        print(f"Avg. validation loss is {validate_loss}.")
        print(f"Total validation time is {end_v_time-start_v_time} sec.")

        # log metrics
        if args.use_wandb:
            # get losses per channel
            this_time = time.time()
            wandb.log({"iter_time": this_time - iter_start_time,
                       "validation_loss": validate_loss})
            iter_start_time = this_time

        # save the state
        if validate_loss < best_validation:
            best_validation = validate_loss
            if args.verbose:
                print("Saving state.")
            my_model.save(output_path)

        # pass validation loss to scheduler
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {ie}\t \
            Validation Loss: {validate_loss}\t \
            LR:{curr_lr}')
        scheduler.step(validate_loss)

        # record the memory snapshot
        if args.memory_management and use_gpu:
            try:
                torch.cuda.memory._dump_snapshot(f"{memory_file_prefix}.iter_{ie}.pickle")
                print(torch.cuda.memory_summary())
            except Exception as e:
                logger.error(f"Failed to capture memory snapshot {e}")

    # close
    if args.use_wandb:
        wandb.finish()

    # time to train
    train_time = time.time()
    print(f"Total training time: {train_time - start_time} seconds.")

    # Stop recording memory snapshot history.
    if args.memory_management and use_gpu:
        torch.cuda.memory._record_memory_history(enabled=None)

    # collect the garbage to make sure memory profiling is handling it correctly
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This program will train the network and save it.")
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-w', '--use_wandb', action='store_true', help='Use wandb logging.')
    parser.add_argument('-m', '--memory_management', action='store_true', help='Print memory management information.')
    parser.add_argument('-c', '--channelize', action='store_true', help='Channelize the result.')
    parser.add_argument('-1', '--go_to_1x1', action='store_true', help='use layers to get down to 1x1')
    parser.add_argument('-2', '--go_to_2x2', action='store_true', help='use layers to get down to 1x1')
    parser.add_argument('-s', '--seed', type=int, default=17, help='seed for random-number generator')
    parser.add_argument('-n', '--num_epochs', type=int, default=500, help='number of epochs to run')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='number of samples per batch')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('-r', '--reload', type=str, default=None, help='Current model to reload.')
    parser.add_argument('-o', '--output', type=str, default='channelized_model.tar', help='Output name')
    parser.add_argument('-t', '--write_tar', type=str, default=None, help='Write some validation files to tar file.')

    args = parser.parse_args()

    # check parameters
    parse_error = False
    if args.learning_rate < 0 or args.learning_rate > 1:
        parse_error = True
        print(f"Invalid learning rate: {args.learning_rate}.")
    if args.num_epochs <= 0:
        parse_error = True
        print(f"Number of epochs must be positive.")
    if args.batch_size <= 0:
        parse_error = True
        print(f"Batch size must be positive.")
    if parse_error:
        print("Exiting due to parsing errors.")
        sys.exit(1)

    # show verbose information
    if args.verbose:
        print("This program will train the network and save it.")
        print(f"There will be {args.num_epochs} epochs.")
        print(f"The batch size is {args.batch_size}.")
        print(f"The learning rate is {args.learning_rate}.")
        if args.write_tar:
            print(f"Final validations will be written to {args.write_tar}.")
        if args.go_to_1x1:
            print("Will use 1x1 kernels at lowest level.")
        elif args.go_to_2x2:
            print("Will use 2x2 kernels at lowest level.")
        else:
            print("Will use 4x4 kernels at lowest level.")
        print(f"Random seed is {args.seed}.")

    # print the environment
    # import sys
    # import os
    # print(sys.executable)
    # print(os.getcwd())
    # print(sys.path)

    # execute main
    main(args)

    # Done
    print("Done")
