

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


# Taken from Arno Rüegg's implementation of MixMatch(?)
class LabeledUnlabeledSampler(Sampler):
    """
    Samples a batch of labeled and unlabeled data
    """
    def __init__(self, labeled_indices, unlabeled_indices, batch_size):

        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self.batch_size = batch_size

    def __iter__(self):
        """
        Returns an iterator that yields a batch of labeled and unlabeled data
        """
        # Number of labeled and unlabeled data points
        labeled_batch_size = self.batch_size // 2
        unlabeled_batch_size = self.batch_size - labeled_batch_size

        # Sample labeled data points
        labeled_batches = [random.sample(self.labeled_indices, labeled_batch_size) for _ in range(len(self.labeled_indices) // labeled_batch_size)]
        unlabeled_batches = [random.sample(self.unlabeled_indices, unlabeled_batch_size) for _ in range(len(self.unlabeled_indices) // unlabeled_batch_size)]
        
        mixed_batches = labeled_batches + unlabeled_batches
        random.shuffle(mixed_batches)
        return iter(batch for batches in mixed_batches for batch in batches)
    
    def __len__(self):
        return len(self.labeled_indices) + len(self.unlabeled_indices)
    
