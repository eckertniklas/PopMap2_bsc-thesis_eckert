

import torch
import random
from torch.utils.data import Sampler
import torch.utils.data as torch_data





"""
custom_sampler = LabeledUnlabeledSampler(
        labeled_indices=dataset.ind_labeled,
        unlabeled_indices=dataset.ind_unlabeled,
        batch_size=cfg.TRAINER.BATCH_SIZE
    )

dataloader = torch_data.DataLoader(dataset,sampler = custom_sampler, **dataloader_kwargs)
"""


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
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
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
        labeled_batches = [random.sample(self.labeled_indices, labeled_batch_size) for _ in range(length)]
        unlabeled_batches = [random.sample(self.unlabeled_indices, unlabeled_batch_size) for _ in range(length)]
        
        # concatenate labeled and unlabeled indices
        mixed_batches = torch.concat([torch.tensor(labeled_batches), torch.tensor(unlabeled_batches)],1).tolist()

        # yield batches of labeled and unlabeled data
        return iter(batch for batches in mixed_batches for batch in batches)
    
        # yield
    
    # # Resart itaration
    # def __next__(self):
    #     return self.__iter__()
    
    def __len__(self):
        """
        Returns the length of the sampler
        """
        return len(self.labeled_indices)*2 #+ len(self.unlabeled_indices)
    
