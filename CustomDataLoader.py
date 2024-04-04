import torch
#from torch.utils.data import Dataset
import random

class CustomDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Determine the total number of samples in the dataset
        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples))

    def __iter__(self):
        if self.shuffle:
            # Shuffle the indices for each epoch
            random.shuffle(self.indices)

        batch_start = 0
        while batch_start < self.num_samples:
            # Determine the end index of the current batch
            batch_end = min(batch_start + self.batch_size, self.num_samples)
            
            # Get the indices of samples for the current batch
            batch_indices = self.indices[batch_start:batch_end]

            # Get data for the current batch using dataset's __getitem__
            batch_data = [self.dataset[idx] for idx in batch_indices]

            # Stack each tensor separately
            batch_tensors = [torch.stack(tensors) for tensors in zip(*batch_data)]

            # Yield the batch tensor
            yield batch_tensors[0], batch_tensors[1]

            # Move to the next batch
            batch_start += self.batch_size

    def __len__(self):
        # Compute the number of batches
        return (self.num_samples + self.batch_size - 1) // self.batch_size

