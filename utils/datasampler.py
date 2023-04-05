import torch
import random
from torch.utils.data import Sampler
import torch.utils.data as torch_data

# Taken from Arno Rüegg's implementation
class LabeledUnlabeledSampler(Sampler):
    """
    Samples a batch of labeled and unlabeled data
    """
    def __init__(self, labeled_indices, unlabeled_indices, batch_size):
        """
        input:
            labeled_indices: list of indices of labeled data points
            unlabeled_indices: list of indices of unlabeled data points
            batch_size: batch size
        """
        self.labeled_indices = torch.tensor(labeled_indices)
        self.unlabeled_indices = torch.tensor(unlabeled_indices)
        self.batch_size = batch_size

    def __iter__(self):
        """
        Returns an iterator that yields a batch of labeled and unlabeled data
        input:
            None
        output:
            iterator that yields a batch of labeled and unlabeled data
        """
        # Number of labeled and unlabeled data points
        labeled_batch_size = self.batch_size // 2
        unlabeled_batch_size = self.batch_size - labeled_batch_size

        # legth definition
        length = len(self.labeled_indices) // labeled_batch_size

        # Sample labeled data points
        labeled_batches = self.labeled_indices[torch.randperm(len(self.labeled_indices))][:length*labeled_batch_size].view((length, labeled_batch_size))

        # choose (with a multinomial and replacement) just as many unlabeled data points as labeled data points, doing basically over and undersampling on the unlabelled data
        unlabeled_batches = (self.unlabeled_indices[
            torch.multinomial(torch.arange(len(self.unlabeled_indices))/len(self.unlabeled_indices), length*unlabeled_batch_size, replacement=True) ]
            .view((length, unlabeled_batch_size))
        )

        # concatenate labeled and unlabeled indices
        mixed_batches = torch.concat([labeled_batches, unlabeled_batches], 1).tolist()

        # yield batches of labeled and unlabeled data
        return iter(batch for batches in mixed_batches for batch in batches)    
    
    def __len__(self):
        """
        Returns the length of the sampler
        """
        return len(self.labeled_indices)*2 #+ len(self.unlabeled_indices)
    
