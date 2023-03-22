

import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import json

from utils.constants import pop_map_root, config_path

def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


class Population_Dataset_target(Dataset):
    """
    Population dataset for target domain
    Use this dataset to evaluate the model on the target domain and compare it the census data
    """
    def __init__(self, region, S1=False, S2=True, VIIRS=True, patchsize=1024, overlap=32 ) -> None:
        super().__init__()
        self.region = region
        self.S1 = S1
        self.S2 = S2
        self.VIIRS = VIIRS
        self.patchsize = patchsize

        # get the path to the data
        region_root = os.path.join(pop_map_root, region)

        self.boundary_file = os.path.join(region_root, "boundaries4.tif")
        self.census_file = os.path.join(region_root, "census.csv")

        # get the shape of the images
        with rasterio.open(self.boundary_file, "r") as src:
            self.img_shape = src.shape
            self._meta = src.meta.copy()

        # get a list of indices of the possible patches
        self.patch_indices = self.get_patch_indices(patchsize, overlap)

        # get the path to the data


        # load the dataset stats
        self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats.json'))
        for mkey in self.dataset_stats.keys():
            if isinstance(self.dataset_stats[mkey], dict):
                for key,val in self.dataset_stats[mkey].items():
                    self.dataset_stats[mkey][key] = np.array(val)
            else:
                self.dataset_stats[mkey] = np.array(val)

    def get_patch_indices(self, patchsize, overlap):
        """
        :param patchsize: size of the patch
        :param overlap: overlap between patches
        :return: list of indices of the patches
        """
        stride = patchsize - overlap*2
        h,w = self.img_shape

        x = torch.arange(0, h-patchsize, stride)
        y = torch.arange(0, w-patchsize, stride)

        main_indices = torch.cartesian_prod(x,y)

        # also cover the boarder pixels that are not covered by the main indices
        max_x = h-patchsize
        max_y = w-patchsize
        bottom_indices = torch.stack([torch.ones(len(y))*max_x, y]).T
        right_indices = torch.stack([x, torch.ones(len(x))*max_y]).T

        # add the bottom right corner
        bottom_right_idx = torch.tensor([max_x, max_y]).unsqueeze(0)

        return torch.cat([main_indices, bottom_indices, right_indices, bottom_right_idx])

    def metadata(self):
        return self._meta

    def shape(self):
        return self.img_shape

    def __len__(self) -> int:
        return len(self.patch_indices)

    def __getitem__(self, index: int):

        # get the indices of the patch
        x,y = self.patch_indices[index]
        data = self.data_generation(x,y)
        data = torch.from_numpy(data).type(torch.FloatTensor)
        
        return {'input': data, 'img_coords': (x,y)}
    

    def data_generation(self, x,y):
        """
        Generate the data for the patch
        :return: the data
        """
        data = []
        if self.S1:
            with rasterio.open(self.S1_file, "r") as src:
                data = src.read(window=((x,x+self.patchsize),(y,y+self.patchsize)))
            raw_data = self.generate_data(self.S1_file, x, y)
            new_arr = ((raw_data.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
            data.append(new_arr)
        if self.S2:
            with rasterio.open(self.S2_file, "r") as src:
                data = src.read(window=((x,x+self.patchsize),(y,y+self.patchsize)))
            raw_data = self.generate_data(self.S2_file, x, y)
            new_arr = ((raw_data.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
            data.append(new_arr)
        if self.VIIRS:
            with rasterio.open(self.VIIRS_file, "r") as src:
                data = src.read(window=((x,x+self.patchsize),(y,y+self.patchsize)))
            raw_data = self.generate_data(self.VIIRS_file, x, y)
            raw_data = np.where(raw_data < 0, 0, raw_data)
            data.append(new_arr)

        return np.concatenate(data, axis=0)


    def map_to_census(self, pred):
        """
        Map the predicted population to the census data
        :param pred: predicted population
        :return: the mapped population
        """
        pass