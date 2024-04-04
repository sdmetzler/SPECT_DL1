# pull in the trainer module
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import os
import SPECT_Dataset_Channels
import SPECT_Model_channelized
import numpy as np
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from memory_profiler import profile
import gc

def channelizing_test(verbose):
    values = torch.linspace(0., 256 * 256 - 1, steps=256 * 256, requires_grad=True)
    tensor = values.reshape(1, 256, 256)
    multiplied_tensors = []

    # Multiply the original tensor by (i+1) and store the results in the list
    for i in range(10):
        multiplied_tensor = tensor * (i + 1)
        multiplied_tensors.append(multiplied_tensor.unsqueeze(0))

    test_image = torch.cat(multiplied_tensors, dim=0).to(torch.float64)  # [10, 1, 256, 256]

    # check values
    test_eps = 0.01
    pass1 = True
    for i1 in range(10):
        for ix in range(256):
            for jx in range(256):
                expected_value = (i1 + 1) * (256 * jx + ix)
                value = test_image[i1, 0, jx, ix]
                if abs(value - expected_value) > test_eps:
                    print(f"Error in test[{i1}, 1, {jx}, {ix}]. Expected {expected_value}, but found {value}.")
                    pass1 = False

    if verbose:
        if pass1:
            print("Pass 1 of channelizing succeeded.")
        else:
            print("Pass 1 of channelizing failed.")

    # channelize
    pass2 = True
    channelized_image = SPECT_Dataset_Channels.channelize_tensor(test_image)
    if len(channelized_image.shape) != 4:
        print(f"Wrong number of dimensions: {len(channelized_image.shape)}")
        pass2 = False
    if channelized_image.shape[0] != 10:
        print(f"Wrong number of instances: {channelized_image.shape[0]}")
        pass2 = False
    if channelized_image.shape[1] != 9:
        print(f"Wrong number of channels: {channelized_image.shape[1]}")
        pass2 = False
    if channelized_image.shape[-2:] != (256, 256):
        print(f"Wrong image size: {channelized_image.shape[-2]}")
        pass2 = False

    if verbose:
        if pass2:
            print("Pass 2 of channelizing succeeded.")
        else:
            print("Pass 2 of channelizing failed.")

    # check to see if values of higher instances are as expected
    pass3 = True
    for i1 in range(10):
        for i_channel in range(9):
            for jx in range(256):
                for ix in range(256):
                    value = channelized_image[i1, i_channel, jx, ix]
                    this_exp = (i1 + 1) * channelized_image[0, i_channel, jx, ix]
                    if abs(value - this_exp) > test_eps:
                        print(
                            f"Error during pass 3 in test[{i1}, {i_channel}, {jx}, {ix}]. Expected {this_exp}, but found {value}.")
                        pass3 = False

    if verbose:
        if pass3:
            print("Pass 3 of channelizing succeeded.")
        else:
            print("Pass 3 of channelizing failed.")

    # check values of first instance
    pass4 = True
    previous_values = torch.zeros(256, 256)
    for i_channel in range(9):
        k = 256 // int(2 ** i_channel)
        next_values = torch.zeros(256, 256)
        for jx in range(256):
            jxp = jx // k
            for ix in range(256):
                ixp = ix // k
                expected_value = (2 * ixp * k + k - 1) / 2 + 256 * (2 * jxp * k + k - 1) / 2
                next_values[jx, ix] = expected_value
                this_exp = expected_value - previous_values[jx, ix]
                value = channelized_image[0, i_channel, jx, ix]
                if abs(value - this_exp) > test_eps:
                    print(
                        f"Error during pass 4 in test[0, {i_channel}, {jx}, {ix}]. Expected {this_exp}, but found {value}.")
                    pass4 = False
        # save previous
        previous_values = next_values

    if verbose:
        if pass4:
            print("Pass 4 of channelizing succeeded.")
        else:
            print("Pass 4 of channelizing failed.")

    pass5 = True
    dechannelized_image = SPECT_Dataset_Channels.dechannelize(channelized_image)
    if len(dechannelized_image.shape) != 4:
        print(f"Wrong number of dimensions: {len(dechannelized_image.shape)}")
        pass5 = False
    if dechannelized_image.shape[0] != 10:
        print(f"Wrong number of instances: {dechannelized_image.shape[0]}")
        pass5 = False
    if dechannelized_image.shape[1] != 1:
        print(f"Wrong number of channels: {dechannelized_image.shape[1]}")
        pass5 = False
    if dechannelized_image.shape[-2:] != (256, 256):
        print(f"Wrong image size: {dechannelized_image.shape[-2]}")
        pass5 = False

    if verbose:
        if pass5:
            print("Pass 5 of channelizing succeeded.")
        else:
            print("Pass 5 of channelizing failed.")

    pass6 = True
    for i1 in range(10):
        for ix in range(256):
            for jx in range(256):
                expected_value = (i1 + 1) * (256 * jx + ix)
                value = dechannelized_image[i1, 0, jx, ix]
                if abs(value - expected_value) > test_eps:
                    print(f"Error in test[{i1}, 1, {jx}, {ix}]. Expected {expected_value}, but found {value}.")
                    pass6 = False

    if verbose:
        if pass6:
            print("Pass 6 of channelizing succeeded.")
        else:
            print("Pass 6 of channelizing failed.")

    # return result
    return pass1 and pass2 and pass3 and pass4 and pass5 and pass6


@profile
def main(args):
    """
    # create a test image
    channelizing_ok = channelizing_test(args.verbose)
    if not channelizing_ok:
        print("Channelizing test failed.")
        return
    elif args.verbose:
        print("Channelizing test succeeded.")
    """

    use_gpu = torch.cuda.is_available()
    if args.verbose:
        if use_gpu:
            print("GPUs will be used.")
        else:
            print("CPU only.")

    # get the device
    device = torch.device("cuda" if use_gpu else "cpu")

    # start recording memory snapshots
    if args.memory_management and use_gpu:
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    # Prefix for file names.
    TIME_FORMAT_STR = "%Y-%m-%d %H:%M:%S"
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    memory_file_prefix = f"memory_snapshot_{args.go_to_2x2}_{args.extract_at_end}_{timestamp}"

    # spect_path
    spect_path = '/Users/metzler/Work/Python/SPECT_DL/'
    #spect_path = '/home/metzler/NN_Test1/SPECT_DL/'
    #spect_path = '/projects/bcps/smetzler/NN_Trial1/SPECT_DL/'
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
    dataset = SPECT_Dataset_Channels.SPECT_Dataset_Channels(proj_path, '.atten.noisy.proj',
                                                            phantom_path, '.phantom', num_sets,
                                                            normalize_input=True, normalize_label=True)

    # put aside data for testing
    train_batch_size = args.batch_size
    validate_batch_size = args.batch_size
    development_size = int(0.8 * num_sets)
    test_size = num_sets - development_size
    train_size = (development_size * 4) // 5
    validate_size = development_size - train_size
    development_dataset, testing_dataset = random_split(dataset, [development_size, test_size])

    # Create data loader
    train_dataset, validate_dataset = random_split(development_dataset, [train_size, validate_size])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=validate_batch_size, shuffle=True)

    # create the model
    my_model = SPECT_Model_channelized.SPECT_Model_channelized(args.extract_at_end, args.go_to_2x2, args.go_to_1x1)
    if args.reload is not None:
        print(f"Reloading model from: {spect_path + args.reload}")
        my_model = my_model.load_and_replace(spect_path + args.reload)
    my_model = my_model.to(device)
    criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(my_model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)

    # scaler allows mixed precision
    scaler = GradScaler() if use_gpu else None

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
    train_losses = []
    validate_losses = []
    best_validation = 1_000_000.
    for ie in range(args.num_epochs):
        # print info
        print(f"Beginning epoch {ie}.")

        # Train
        #train_mse = 0.
        train_loss = 0.
        for batch, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            with autocast():  # Enables automatic mixed precision
                y_pred = my_model.forward(X_train)
                loss = criterion(y_pred, y_train)
                #mse_per_channel = torch.mean((y_train - y_pred) ** 2, dim=(0, 2, 3))
                #train_mse += mse_per_channel
                train_loss += loss

            # print summary
            print(f"Training loss for batch {batch} of epoch {ie} is {loss}.")
            train_losses.append(loss)

            # update our parameters
            optimizer.zero_grad()
            if use_gpu:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # update our parameters
                loss.backward()
                optimizer.step()
                train_losses.append(loss)
            optimizer.zero_grad()

            # Release intermediate variables
            #del loss, mse_per_channel
            del loss
            if use_gpu:
                torch.cuda.empty_cache()

            break

        # get averages
        #train_mse /= (batch + 1)
        train_loss /= (batch + 1)
        print(f"Avg. training loss is {train_loss}.")
        #print(f"Avg. training loss per channel is: {train_mse}")

        # Validate
        #validate_mse = 0.
        validate_loss = 0.
        with torch.no_grad():
            for batch, (X_validate, y_validate) in enumerate(validate_loader):
                X_validate = X_validate.to(device)
                y_validate = y_validate.to(device)
                y_pred = my_model(X_validate)
                loss = criterion(y_pred, y_validate)
                validate_losses.append(loss)
                #mse_per_channel = torch.mean((y_validate - y_pred) ** 2, dim=(1, 2))
                #validate_mse += mse_per_channel
                validate_loss += loss

                # Release intermediate variables
                #del loss, mse_per_channel
                del loss
                if use_gpu:
                    torch.cuda.empty_cache()

                break

            # get averages
            #validate_mse /= (batch + 1)
            validate_loss /= (batch + 1)
            print(f"Avg. validation loss is {validate_loss}.")
            #print(f"Avg. validation loss per channel is: {validate_mse}")

        # log metrics
        if args.use_wandb:
            # get losses per channel
            this_time = time.time()
            #wandb.log({"iter_time": this_time - iter_start_time, "training_loss": train_loss,
            #           "training_loss_channel": train_mse, "validation_loss": validate_loss,
            #           "validation_loss_channel": validate_mse})
            wandb.log({"iter_time": this_time - iter_start_time, "training_loss": train_loss,
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
            Training Loss: {train_loss}\t \
            Validation Loss:{validate_loss}\t \
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
    parser.add_argument('-e', '--extract_at_end', action='store_true',
                        help='Extract the channelized parameters from the final image only.')
    parser.add_argument('-w', '--use_wandb', action='store_true', help='Use wandb logging.')
    parser.add_argument('-m', '--memory_management', action='store_true', help='Print memory management information.')
    parser.add_argument('-1', '--go_to_1x1', action='store_true', help='use layers to get down to 1x1')
    parser.add_argument('-2', '--go_to_2x2', action='store_true', help='use layers to get down to 1x1')
    parser.add_argument('-s', '--seed', type=int, default=7, help='seed for random-number generator')
    parser.add_argument('-n', '--num_epochs', type=int, default=500, help='number of epochs to run')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='number of samples per batch')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('-r', '--reload', type=str, default=None, help='Current model to reload.')
    parser.add_argument('-o', '--output', type=str, default='channelized_model.tar', help='Output name')

    args = parser.parse_args()

    args.extract_at_end = True
    args.num_epochs = 1
    args.verbose = True

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
        print(f"The learning rate is {args.learning_rate}.")
        if args.extract_at_end:
            print("Channel extraction will be at the end of epoch.")
        else:
            print("Channel extraction will occur during forward.")
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
