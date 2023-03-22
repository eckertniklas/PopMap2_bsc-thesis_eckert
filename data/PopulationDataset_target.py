

import os
from torch.utils.data import Dataset

from utils.constants import pop_map_root


class Population_Dataset_target(Dataset):
    """
    Population dataset for target domain
    Use this dataset to evaluate the model on the target domain and compare it the census data
    """
    def __init__(self, region ) -> None:
        super().__init__()
        self.region = region

        # get the path to the data
        region_root = os.path.join(pop_map_root, region)

        # get list of samples for the region (only spring)
        self.samples = os.listdir(region_root, "sen2spring")

        # get the path to the data



    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int):
        pass