# pull in the trainer module
import argparse
import gc
import random
import sys
import time
from datetime import datetime

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

import CustomDataLoader
import SPECT_Dataset5
import SPECT_Model_channelized2
import pickle

# settings
num_sets = 2000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_loss(the_model, the_loader, the_criterion, write, write_path):
    the_loss = 0.
    #print(f"the_model: {the_model}")
    #the_model.save("get_loss.model.tar")
    #with open('mse_loss_function.pkl', 'wb') as f:
    #    pickle.dump(the_criterion, f)
    with torch.no_grad():
        for batch, (X, y) in enumerate(the_loader):
            #print(f"Have data: {X.shape}.")
            y_pred = the_model(X)
            loss = the_criterion(y_pred, y)
            #print(f"Loss for batch {batch} is {loss}.")
            assert not torch.isnan(loss), "Loss is NaN. Exiting."
            the_loss += loss

            # print more info
            #for ii in range(X.shape[0]):
            #    print(f"data {ii}: {torch.mean(X[ii, :, :, :])} {torch.mean(y[ii, :, :, :])} {torch.mean(y_pred[ii, :, :, :])}.")

        # get averages
        the_loss /= (batch + 1)

    # see if should write
    if write:
        print("Writing tar files")
        fname_x = write_path + '_x.tar'
        fname_y = write_path + '_y.tar'
        fname_p = write_path + '_p.tar'
        torch.save(X, fname_x)
        torch.save(y, fname_y)
        torch.save(y_pred, fname_p)
        print("Write complete.")

    # return result
    return the_loss

def randomly_split_sets(n_sets, s1, s2, s3):
    if n_sets != s1 + s2 + s3:
        raise ValueError("Total number of sets must equal to the sum of s1, s2, and s3.")

    # Create a list containing numbers from 0 to n_sets - 1
    all_sets = list(range(n_sets))

    # Shuffle the list to randomize the order of sets
    random.shuffle(all_sets)

    # Split the shuffled list into mutually exclusive sets of sizes s1, s2, and s3
    set1 = all_sets[:s1]
    set2 = all_sets[s1:s1 + s2]
    set3 = all_sets[s1 + s2:s1 + s2 + s3]
    assert len(set1) + len(set2) + len(set3) == n_sets

    return set1, set2, set3


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
    spect_path_access = '/projects/bcps/smetzler/NN_Trial1/SPECT_DL/'
    spect_path = spect_path_access if use_gpu else spect_path_laptop
    proj_path = spect_path + "TrainData/AttenProj/"
    phantom_path = spect_path + "TrainData/Phantoms/"
    print(f"Data path is {spect_path}.")

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
                "learning_rate": args.learning_rate,
                "architecture": "CNN",
                "dataset": "SPECT_TRAIN",
                "epochs": args.num_epochs,
            }
        )

    # set a random seed for reproducibility
    torch.manual_seed(args.seed)

    # put aside data for testing
    train_size = (num_sets * 4) // 5
    validate_size = num_sets // 5
    test_size = num_sets - train_size - validate_size
    print(f"There are {num_sets} datasets, with a train size of {train_size}.")
    print(f"The validate and test sizes are {validate_size} and {test_size}.")
    # for now test_size is set to zero; num_sets is also set to 2000. This way I reserve
    # the last 500 for testing.
    assert test_size >= 0
    #train_set, validate_set, test_set = randomly_split_sets(num_sets, train_size, validate_size, test_size)
    train_set = list(range(train_size))
    #print(f"train_set: {train_set}")
    assert len(train_set) == train_size
    validate_set = list(range(train_size, num_sets))
    #print(f"validate_set: {validate_set}")
    assert len(validate_set) == validate_size
    test_set = list(range(num_sets, num_sets))
    #print(f"test_set: {test_set}")
    assert len(test_set) == test_size

    # load the data
    # Create custom datasets
    start_d_time = time.time()
    proj_type = '.atten.noiseless.proj'
    normalize_input = False
    add_noise = False
    print(f"Normalize input: {normalize_input}.")
    print(f"Add noise: {add_noise}.")
    train_dataset = SPECT_Dataset5.SPECT_Dataset5(proj_path, proj_type,
                                                  phantom_path, '.phantom', train_set,
                                                  normalize_input=normalize_input, add_noise=add_noise)
    assert train_dataset.__len__() == train_size
    validate_dataset = SPECT_Dataset5.SPECT_Dataset5(proj_path, proj_type,
                                                     phantom_path, '.phantom', validate_set,
                                                     normalize_input=normalize_input, add_noise=add_noise)
    assert validate_dataset.__len__() == validate_size
    test_dataset = SPECT_Dataset5.SPECT_Dataset5(proj_path, proj_type,
                                                 phantom_path, '.phantom', test_set,
                                                 normalize_input=normalize_input, add_noise=add_noise)
    assert test_dataset.__len__() == test_size
    end_d_time = time.time()
    print(f"Dataset creating time: {end_d_time - start_d_time} sec.")

    # expand sets
    train_dataset.expand_data(args.expansion)
    assert train_dataset.__len__() == train_size * args.expansion
    validate_dataset.expand_data(args.expansion)
    assert validate_dataset.__len__() == validate_size * args.expansion
    test_dataset.expand_data(args.expansion)
    assert test_dataset.__len__() == test_size * args.expansion
    end_de_time = time.time()
    print(f"Dataset was expanded by a factor of {args.expansion}.")
    print(f"Dataset expansion time: {end_de_time - end_d_time} sec.")

    # set the batch sizes
    train_batch_size = min(args.batch_size, train_size * args.expansion)
    validate_batch_size = min(args.batch_size, validate_size * args.expansion)
    print(f"Train and validate batch sizes are {train_batch_size} and {validate_batch_size}.")

    # Create data loader
    train_loader = CustomDataLoader.CustomDataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    validate_loader = CustomDataLoader.CustomDataLoader(dataset=validate_dataset, batch_size=validate_batch_size,
                                                        shuffle=False)
    test_batch_size = min(test_size * args.expansion, validate_batch_size)
    test_loader = CustomDataLoader.CustomDataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    # create the model
    my_model = SPECT_Model_channelized2.SPECT_Model_channelized2(args.channelize, args.go_to_2x2, args.go_to_1x1)
    if args.reload is not None:
        if args.reload[0] == '/':
            print(f"Reloading model from: {args.reload}")
            my_model = my_model.load_and_replace(args.reload, device)
        else:
            print(f"Reloading model from: {spect_path + args.reload}")
            my_model = my_model.load_and_replace(spect_path + args.reload, device)
    my_model = my_model.to(device)  # seems like this shouldn't be necessary because device is used in the above
    #  lines, but there was an error suggesting that the data and model were not both on the GPU.

    if args.verbose:
        print(f"Model has {count_parameters(my_model)} parameters.")

    # create the criterion
    criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(my_model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    # scaler allows mixed precision
    # scaler = GradScaler() if use_gpu else None

    # record the current time
    start_time = time.time()
    iter_start_time = start_time

    # record the memory snapshot
    if args.memory_management and use_gpu:
        try:
            torch.cuda.memory._dump_snapshot(f"{memory_file_prefix}.pickle")
        except Exception as e:
            print(f"Failed to capture memory snapshot {e}")

    # get validation loss before optimizing
    print(f"Initial validation loss is {get_loss(my_model, validate_loader, criterion, False, './validate_')}.")

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
            if ie == 0:
                print(f"Training loss for batch {batch} of epoch {ie} is {loss}.")

            # update our parameters
            # start_g_time = time.time()
            optimizer.zero_grad()
            if use_gpu:
                # update our parameters
                loss.backward()
                optimizer.step()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
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

        # Validate
        validate_loss = 0.
        start_v_time = time.time()
        with torch.no_grad():
            for batch, (X_validate, y_validate) in enumerate(validate_loader):
                y_pred = my_model(X_validate)
                loss = criterion(y_pred, y_validate)
                assert not torch.isnan(loss), "Validation loss is NaN. Exiting."
                validate_loss += loss

            # get averages
            validate_loss /= (batch + 1)
        end_v_time = time.time()
        print(f"Avg. validation loss is {validate_loss}.")
        print(f"Total validation time is {end_v_time - start_v_time} sec.")

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
                print(f"Failed to capture memory snapshot {e}")

    # close
    if args.use_wandb:
        wandb.finish()

    # time to train
    train_time = time.time()
    print(f"Total training time: {train_time - start_time} seconds.")

    # Test
    test_loss = 0.
    start_test_time = time.time()
    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_loader):
            y_pred = my_model(X_test)
            loss = criterion(y_pred, y_test)
            assert not torch.isnan(loss), "Test loss is NaN. Exiting."
            test_loss += loss

            if args.write_tar:
                fname_x = spect_path + args.write_tar + '_x.tar'
                fname_y = spect_path + args.write_tar + '_y.tar'
                fname_p = spect_path + args.write_tar + '_p.tar'
                if args.verbose:
                    print(f"Writing validation to files {fname_x}, {fname_y}, and {fname_p}.")
                    print(f"Shapes are {X_validate.shape}, {y_validate.shape}, and {y_pred.shape}")
                torch.save(X_test, fname_x)
                torch.save(y_test, fname_y)
                torch.save(y_pred, fname_p)

        # get averages
        test_loss /= (batch + 1)
        end_test_time = time.time()
        print(f"Avg. test loss is {test_loss}.")
        print(f"There were {batch + 1} test batches.")

    # if not writing files, store indices
    if not args.write_tar:
        print(f"Test set is: {test_set}")

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
    parser.add_argument('-b', '--batch_size', type=int, default=int(num_sets * 4) // 25,
                        help='number of samples per batch')
    parser.add_argument('-e', '--expansion', type=int, default=1, help='data size expansion using sinogram rotation')
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
    if args.expansion <= 0:
        parse_error = True
        print(f"Expansion size must be positive.")
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
