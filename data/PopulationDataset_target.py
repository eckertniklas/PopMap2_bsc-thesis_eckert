

import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy import ndimage
from scipy.interpolate import griddata
import rasterio
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import random

from typing import Dict, Tuple

from utils.constants import pop_map_root_large, pop_map_covariates, pop_map_covariates_large, config_path 
from utils.constants import datalocations
from utils.plot import plot_2dmatrix


def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


class Population_Dataset_target(Dataset):
    """
    Population dataset for target domain
    Use this dataset to evaluate the model on the target domain and compare it the census data
    """
    def __init__(self, region, S1=False, S2=True, VIIRS=True, NIR=False, patchsize=1024, overlap=32, fourseasons=False, mode="test",
                 max_samples=None, transform=None) -> None:
        """
        Input:
            region: the region identifier (e.g. "pri" for puerto rico)
            S1: whether to use sentinel 1 data
            S2: whether to use sentinel 2 data
            VIIRS: whether to use VIIRS data
            NIR: whether to use NIR data
            patchsize: the size of the patches to extract
            overlap: the overlap between patches
            fourseasons: whether to use the four seasons data
            mode: the mode to use ("weaksup" (weakly supervised training) or "test")
        """
        super().__init__()

        self.region = region
        self.S1 = S1
        self.S2 = S2
        self.NIR = NIR
        self.VIIRS = VIIRS
        self.patchsize = patchsize
        self.overlap = overlap
        self.fourseasons = fourseasons
        self.mode = mode
        self.transform = transform
        self.use2A = True

        # get the path to the data
        region_root = os.path.join(pop_map_root_large, region)

        # load the boundary and census data
        levels = datalocations[region].keys()
        self.file_paths = {}
        for level in levels:
            self.file_paths[level] = {}
            for data_type in ["boundary", "census"]:
                self.file_paths[level][data_type] = os.path.join(region_root, datalocations[region][level][data_type])

            self.boundary_file = os.path.join(region_root, datalocations[region]["fine"]["boundary"])
            self.census_file = os.path.join(region_root, datalocations[region]["fine"]["census"])
            self.coarse_boundary_file = os.path.join(region_root, datalocations[region]["coarse"]["boundary"])
            self.coarse_census_file = os.path.join(region_root, datalocations[region]["coarse"]["census"])
            
        # weaksup data specific preparation
        if self.mode == "weaksup":
            # read the census file
            self.coarse_census = pd.read_csv(self.file_paths["coarse"]["census"])
            # self.coarse_census = pd.read_csv(self.coarse_census_file)
            max_pix = 2e6
            print("Kicking out ", (self.coarse_census["count"]>=max_pix).sum(), "samples with more than ", int(max_pix), " pixels")
            self.coarse_census = self.coarse_census[self.coarse_census["count"]<max_pix].reset_index(drop=True)

            # redefine indexing
            if max_samples is not None:
                self.coarse_census = self.coarse_census.sample(frac=1, random_state=1610)[:max_samples].reset_index(drop=True)
            print("Using", len(self.coarse_census), "samples for weakly supervised training")

            # get the shape of the coarse regions
            with rasterio.open(self.coarse_boundary_file, "r") as src:
                self.cr_regions = src.read(1)

        elif self.mode=="test":
            # testdata specific preparation
            # get the shape and metadata of the images
            with rasterio.open(self.file_paths[list(self.file_paths.keys())[0]]["boundary"], "r") as src:
                self.img_shape = src.shape
                self._meta = src.meta.copy()
            self._meta.update(count=1, dtype='float32', nodata=None)

            # get a list of indices of the possible patches
            self.patch_indices = self.get_patch_indices(patchsize, overlap)
        else:
            raise ValueError("Mode not recognized")

        # get the path to the data files
        covar_root = os.path.join(pop_map_covariates, region)
        # covar_root = os.path.join(pop_map_covariates_large, region)
        # self.S1_file = os.path.join(covar_root,  os.path.join("S1", region +"_S1.tif"))
        self.S1spring_file = os.path.join(covar_root,  os.path.join("S1spring", region +"_S1spring.tif"))
        self.S1summer_file = os.path.join(covar_root,  os.path.join("S1summer", region +"_S1summer.tif"))
        self.S1autumn_file = os.path.join(covar_root,  os.path.join("S1autumn", region +"_S1autumn.tif"))
        self.S1winter_file = os.path.join(covar_root,  os.path.join("S1winter", region +"_S1winter.tif"))
        self.S1_file = {0: self.S1spring_file, 1: self.S1summer_file, 2: self.S1autumn_file, 3: self.S1winter_file}
        # self.S2spring_file = os.path.join(covar_root,  os.path.join("S2spring", region +"_S2spring.tif"))
        # self.S2summer_file = os.path.join(covar_root,  os.path.join("S2summer", region +"_S2summer.tif"))
        # self.S2autumn_file = os.path.join(covar_root,  os.path.join("S2autumn", region +"_S2autumn.tif"))
        # self.S2winter_file = os.path.join(covar_root,  os.path.join("S2winter", region +"_S2winter.tif"))
        if self.use2A:
            self.S2spring_file = os.path.join(covar_root,  os.path.join("S2Aspring", region +"_S2Aspring.tif"))
            self.S2summer_file = os.path.join(covar_root,  os.path.join("S2Asummer", region +"_S2Asummer.tif"))
            self.S2autumn_file = os.path.join(covar_root,  os.path.join("S2Aautumn", region +"_S2Aautumn.tif"))
            self.S2winter_file = os.path.join(covar_root,  os.path.join("S2Awinter", region +"_S2Awinter.tif"))
        else:
            self.S2spring_file = os.path.join(covar_root,  os.path.join("S21Cspring", region +"_S21Cspring.tif"))
            self.S2summer_file = os.path.join(covar_root,  os.path.join("S21Csummer", region +"_S21Csummer.tif"))
            self.S2autumn_file = os.path.join(covar_root,  os.path.join("S21Cautumn", region +"_S21Cautumn.tif"))
            self.S2winter_file = os.path.join(covar_root,  os.path.join("S21Cwinter", region +"_S21Cwinter.tif"))
        self.S2_file = {0: self.S2spring_file, 1: self.S2summer_file, 2: self.S2autumn_file, 3: self.S2winter_file}
        self.season_dict = {0: "spring", 1: "summer", 2: "autumn", 3: "winter"}
        self.inv_season_dict = {v: k for k, v in self.season_dict.items()}
        self.VIIRS_file = os.path.join(covar_root,  os.path.join("viirs", region +"_viirs.tif"))

        # normalize the dataset (do not use, this does not make sense for variable regions sizes like here)
        self.y_stats = load_json(os.path.join(config_path, 'dataset_stats', 'label_stats.json'))
        self.y_stats['max'] = float(self.y_stats['max'])
        self.y_stats['min'] = float(self.y_stats['min'])

        # load the dataset stats
        # if self.use2A:
        #     self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats_unified_2A.json'))
        # else:
        #     self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats_unified.json'))
        # for mkey in self.dataset_stats.keys():
        #     if isinstance(self.dataset_stats[mkey], dict):
        #         for key,val in self.dataset_stats[mkey].items():
        #             self.dataset_stats[mkey][key] = torch.tensor(val)
        #     else:
        #         self.dataset_stats[mkey] = torch.tensor(val)

    def get_patch_indices(self, patchsize, overlap):
        """
        :param patchsize: size of the patch
        :param overlap: overlap between patches
        :return: list of indices of the patches
        """
        # get the indices of the main patches
        stride = patchsize - overlap*2
        h,w = self.img_shape

        # get the indices of the main patches
        x = torch.arange(0, h-patchsize, stride, dtype=int)
        y = torch.arange(0, w-patchsize, stride, dtype=int)
        main_indices = torch.cartesian_prod(x,y)

        # also cover the boarder pixels that are not covered by the main indices
        max_x = h-patchsize
        max_y = w-patchsize
        bottom_indices = torch.stack([torch.ones(len(y), dtype=int)*max_x, y]).T
        right_indices = torch.stack([x, torch.ones(len(x), dtype=int)*max_y]).T

        # add the bottom right corner
        bottom_right_idx = torch.tensor([max_x, max_y]).unsqueeze(0)

        # concatenate all the indices
        main_indices = torch.cat([main_indices, bottom_indices, right_indices, bottom_right_idx])

        # concatenate the season indices, e.g encode the season an enlongate the indices
        season_template = torch.ones(main_indices.shape[0], dtype=int).unsqueeze(1)
        if self.fourseasons:
            main_indices = torch.cat([
                torch.cat([main_indices, season_template*0], dim=1),
                torch.cat([main_indices, season_template*1], dim=1),
                torch.cat([main_indices, season_template*2], dim=1),
                torch.cat([main_indices, season_template*3], dim=1)],
                dim=0
            )
        else:
            main_indices = torch.cat([main_indices, season_template*0], dim=1)

        return main_indices


    def metadata(self):
        return self._meta

    def shape(self) -> Tuple[int, int]:
        return self.img_shape

    def __len__(self) -> int:
        if self.mode=="test":
            return len(self.patch_indices)
        elif self.mode=="weaksup":
            return len(self.coarse_census)

    def __getitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        """
        Description:
            Get the item at the given index, depending on the mode, the item is either a patch or a coarse region,
        Input:
            index: index of the item
        Output:
            item: dictionary containing the input, the mask, the coordinates of the patch, the season and the season string
        """
        if self.mode=="test":
            return self.__gettestitem__(index)
        elif self.mode=="weaksup":
            return self.__getadminitem__(index)


    def __getadminitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        # get the indices of the patch
        census_sample = self.coarse_census.loc[index]
        
        # get the coordinates of the patch
        xmin, xmax, ymin, ymax = tuple(map(int, census_sample["bbox"].strip('()').split(',')))

        # get the season for the S2 data
        season = random.choice(['spring', 'autumn', 'winter', 'summer']) if self.fourseasons else "spring"

        # get the data
        indata, auxdata = self.generate_raw_data(xmin, ymin, self.inv_season_dict[season], patchsize=(xmax-xmin, ymax-ymin), overlap=0)

        if "S2" in indata:
            if np.any(np.isnan(indata["S2"])): 
                indata["S2"] = self.interpolate_nan(indata["S2"]) 
            
        if "S1" in indata:
            if np.any(np.isnan(indata["S1"])):
                indata["S1"] = self.interpolate_nan(indata["S1"]) 

        # get admin_mask
        admin_mask = torch.from_numpy(self.cr_regions[xmin:xmax, ymin:ymax]==census_sample["idx"])

        # To Torch
        indata = {key:torch.from_numpy(np.asarray(val, dtype=np.float32)).type(torch.FloatTensor) for key,val in indata.items()}

        # Modality wise transformations
        # if self.transform:
        #     if "S2" in self.transform and "S2" in indata:
        #         indata["S2"] = self.transform["S2"](indata["S2"])
        #     if "S1" in self.transform and "S1" in indata:
        #         indata["S1"] = self.transform["S1"](indata["S1"])

        # Normalization
        # indata = self.normalize_indata(indata)

        # merge inputs
        # X = torch.concatenate([indata[key] for key in ["S2", "S1", "VIIRS"] if key in indata], dim=0)

        # General transformations
        # if self.transform:
        #     if "general" in self.transform:
        #         X, admin_mask = self.transform["general"]((X, admin_mask.unsqueeze(0)))
        #         admin_mask = admin_mask.squeeze(0)

        # return dictionary
        return {
                # 'input': X,
                **indata,
                'y': torch.from_numpy(np.asarray(census_sample["POP20"])).type(torch.FloatTensor),
                'admin_mask': admin_mask,
                'img_coords': (xmin,ymin), 'valid_coords':  (xmin, xmax, ymin, ymax),
                'season': self.inv_season_dict[season],# 'season_str': [season],
                'source': torch.tensor(True), "census_idx": torch.tensor([census_sample["idx"]]),
                }


    def __gettestitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        # get the indices of the patch
        x,y,season = self.patch_indices[index]

        # get the data
        indata, mask = self.generate_raw_data(x,y,season.item())

        if "S2" in indata:
            if np.any(np.isnan(indata["S2"])): 
                indata["S2"] = self.interpolate_nan(indata["S2"]) 
            
        if "S1" in indata:
            if np.any(np.isnan(indata["S1"])):
                indata["S1"] = self.interpolate_nan(indata["S1"]) 

        # To Torch
        indata = {key:torch.from_numpy(np.asarray(val, dtype=np.float32)).type(torch.FloatTensor) for key,val in indata.items()} 
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        # get valid coordinates of the patch
        xmin = x+self.overlap
        xmax = x+self.patchsize-self.overlap
        ymin = y+self.overlap
        ymax = y+self.patchsize-self.overlap

        # transform the input data with augmentations
        # if self.transform:
        #     if "S2" in self.transform and "S2" in indata:
        #         indata["S2"] = self.transform["S2"](indata["S2"])
        #     if "S1" in self.transform and "S1" in indata:
        #         indata["S1"] = self.transform["S1"](indata["S1"])

        # Normalization
        # indata = self.normalize_indata(indata)

        # merge inputs
        # X = torch.concatenate([indata[key] for key in ["S2", "S1", "VIIRS"] if key in indata], dim=0)

        # if torch.any(torch.isnan(X)):
        #     raise Exception("Nan in X")

        # # General transformations
        # if self.transform:
        #     if "general" in self.transform:
        #         X, mask = self.transform["general"]((X, mask.unsqueeze(0)))
        #         admin_mask = admin_mask.squeeze(0)

        # return dictionary
        return {
                # 'input': X,
                'img_coords': (x,y), 'valid_coords':  (xmin, xmax, ymin, ymax),
                **indata,
                'season': season.item(), 'mask': mask, 'season_str': self.season_dict[season.item()]}
    

    def interpolate_nan(self, input_array):
        """
        Interpolate the NaN values in the input array using bicubic interpolation, extrapolating if necessary using nearest neighbor
        Input:
            input_array: input array with NaN values
        Output:
            input_array: input array with NaN values interpolated
        """

        # Create an array with True values for NaN positions (interpolation mask)
        nan_mask = np.isnan(input_array)
        known_points = np.where(~nan_mask)
        values = input_array[known_points]
        missing_points = np.where(nan_mask)
        # interpolated_values = griddata(known_points[::-1].T, values, missing_points[::-1].T, method='cubic')
        interpolated_values = griddata(np.vstack(known_points).T, values, np.vstack(missing_points).T, method='nearest')

        # fillin the missing ones
        input_array[missing_points] = interpolated_values

        return input_array


    def generate_raw_data(self,x,y, season, patchsize=None, overlap=None):
        """
        Generate the data for the patch
        Input:
            x: x coordinate of the patch
            y: y coordinate of the patch
            season: season of the patch
            patchsize: size of the patch
            overlap: overlap of the patches
        :return:
            data: data of the patch
        """

        patchsize_x = self.patchsize if patchsize is None else patchsize[0]
        patchsize_y = self.patchsize if patchsize is None else patchsize[1]
        overlap = self.overlap if overlap is None else overlap

        indata = {}
        mask = np.zeros((patchsize_x, patchsize_y), dtype=bool)
        mask[overlap:patchsize_x-overlap, overlap:patchsize_y-overlap] = True
        fake = False

        # get the input data
        if self.S2:
            S2_file = self.S2_file[season]
            if self.NIR:
                if fake:
                    indata["S2"] = np.random.randint(0, 10000, size=(4,patchsize_x,patchsize_y))
                else:
                    with rasterio.open(S2_file, "r") as src:
                        # indata["S2"] = src.read((4,3,2,8), window=((x,x+patchsize_x),(y,y+patchsize_y))) 
                        indata["S2"] = src.read((3,2,1,4), window=((x,x+patchsize_x),(y,y+patchsize_y))).astype(np.float32) 
            else:
                if fake:
                    indata["S2"] = np.random.randint(0, 10000, size=(3,patchsize_x,patchsize_y))
                else:
                    with rasterio.open(S2_file, "r") as src:
                        # indata["S2"] = src.read((4,3,2), window=((x,x+patchsize_x),(y,y+patchsize_y)))
                        indata["S2"] = src.read((3,2,1), window=((x,x+patchsize_x),(y,y+patchsize_y))).astype(np.float32) 
            # mask = mask & (indata["S2"].sum(axis=0) != 0)
        if self.S1:
            S1_file = self.S1_file[season]
            if fake:
                indata["S1"] = np.random.randint(0, 10000, size=(2,patchsize_x,patchsize_y))
            else:
                with rasterio.open(S1_file, "r") as src:
                    indata["S1"] = src.read((1,2), window=((x,x+patchsize_x),(y,y+patchsize_y))).astype(np.float32) 
            # mask = mask & (indata["S1"].sum(axis=0) != 0)
        if self.VIIRS:
            if fake:
                indata["VIIRS"] = np.random.randint(0, 10000, size=(1,patchsize_x,patchsize_y))
            else:
                with rasterio.open(self.VIIRS_file, "r") as src:
                    indata["VIIRS"] = src.read(1, window=((x,x+patchsize_x),(y,y+patchsize_y))).astype(np.float32) 
            # mask = mask & (indata["VIIRS"].sum(axis=0) != 0)

        # # load administrative mask
        # admin_mask = self.cr_regions[x:x+patchsize_x, y:y+patchsize_y]==idx

        return indata, mask

    def convert_popmap_to_census(self, pred, gpu_mode=False, level="fine"):
        """
        Converts the predicted population to the census data
        inputs:
            :param pred: predicted population
            :param gpu_mode: if aggregation is done on gpu (can use a bit more GPU memory, but is a lot faster)
        outputs:
            :return: the predicted population for each census region
        """

        boundary_file = self.file_paths[level]["boundary"]
        census_file = self.file_paths[level]["census"]

        # boundary_file = self.boundary_file
        # census_file = self.census_file

        # raise NotImplementedError
        with rasterio.open(boundary_file, "r") as src:
            boundary = src.read(1)
        boundary = torch.from_numpy(boundary)

        # read the census file
        census = pd.read_csv(census_file)

        if gpu_mode:
            pred = pred.cuda()
            boundary = boundary.cuda()
            census_pred = torch.zeros(len(census), dtype=torch.float32).cuda()

            # iterate over census regions and get totals
            # for i, (cidx,bbox) in tqdm(enumerate(zip(census["idx"], census["bbox"])), total=len(census)):
            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
            # for i, (cidx,bbox) in tqdm(enumerate(zip(census["idx"], census["bbox"]))):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                census_pred[i] = pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx].sum()

        else:
            census_pred = torch.zeros(len(census), dtype=torch.float32)

            # iterate over census regions and get totals
            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                # xmin, xmax, ymin, ymax = tuple(map(int, tuple(map(int, bbox.strip('()').strip('[]').split(','))).split(',')))
                census_pred[i] = pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx].sum()
        
        del boundary, pred
        torch.cuda.empty_cache()

        return census_pred, torch.tensor(census["POP20"])
    
    def normalize_indata(self,indata):
        """
        Normalize the input data
        inputs:
            :param indata: input data
        outputs:
            :return: normalized input data
        """

        # S2
        if "S2" in indata:
            if indata["S2"].shape[0] == 4:
                # indata["S2"] = torch.where(indata["S2"] > self.dataset_stats["sen2springNIR"]['p2'][:,None,None], self.dataset_stats["sen2springNIR"]['p2'][:,None,None], indata["S2"])
                indata["S2"] = ((indata["S2"].permute((1,2,0)) - self.dataset_stats["sen2springNIR"]['mean'] ) / self.dataset_stats["sen2springNIR"]['std']).permute((2,0,1))
            else: 
                # indata["S2"] = torch.where(indata["S2"] > self.dataset_stats["sen2spring"]['p2'][:,None,None], self.dataset_stats["sen2spring"]['p2'][:,None,None], indata["S2"])
                indata["S2"] = ((indata["S2"].permute((1,2,0)) - self.dataset_stats["sen2spring"]['mean'] ) / self.dataset_stats["sen2spring"]['std']).permute((2,0,1))

        # S1
        if "S1" in indata:
            # indata["S1"] = torch.where(indata["S1"] > self.dataset_stats["sen1"]['p2'][:,None,None], self.dataset_stats["sen1"]['p2'][:,None,None], indata["S1"])
            indata["S1"] = ((indata["S1"].permute((1,2,0)) - self.dataset_stats["sen1"]['mean'] ) / self.dataset_stats["sen1"]['std']).permute((2,0,1))

        # VIIRS
        if "VIIRS" in indata:
            # indata["VIIRS"] = torch.where(indata["VIIRS"] > self.dataset_stats["viirs"]['p2'][:,None,None], self.dataset_stats["viirs"]['p2'][:,None,None], indata["VIIRS"])
            indata["VIIRS"] = ((indata["VIIRS"].permute((1,2,0)) - self.dataset_stats["viirs"]['mean'] ) / self.dataset_stats["viirs"]['std']).permute((2,0,1))

        return indata
    
    def normalize_reg_labels(self, y): 
        """
        Normalizes the labels to be between 0 and 1
        inputs:
            :param y: the labels
        outputs:
            :return: the normalized labels
        """
        y_max = self.y_stats['max']
        y_min = self.y_stats['min']
        y_scaled = (y - y_min) / (y_max - y_min)
        return y_scaled

    def denormalize_reg_labels(self, y_scaled): 
        """
        Denormalizes the labels back to the original values
        inputs:
            :param y_scaled: the normalized labels
        outputs:
            :return: the denormalized labels
        """ 
        y_max = self.y_stats['max']
        y_min = self.y_stats['min']
        y = y_scaled * (y_max - y_min) + y_min
        return y

    def save(self, preds, output_folder, tag="") -> None:
        """
        Saves the predictions to a tif file
        inputs:
            :param preds: the predictions
            :param output_folder: the folder to save the predictions to (will be created if it doesn't exist)
        outputs:
            :return: None
        """

        # convert to numpy array
        preds = preds.cpu().numpy()

        # create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # save the predictions
        output_file = os.path.join(output_folder, self.region + f"_predictions{tag}.tif")
        with rasterio.open(output_file, "w", **self._meta) as dest:
            dest.write(preds,1)


# collate function for the dataloader
def Population_Dataset_collate_fn(batch):
    """
    Collate function for the dataloader used in the Population_Dataset class
    to ensure that all items in the batch have the same shape
    inputs:
        :param batch: the batch of data with irregular shapes
    outputs:
        :return: the batch of data with same shapes
    """
    # Find the maximum dimensions for each item in the batch
    max_x = max([item['input'].shape[1] for item in batch])
    max_y = max([item['input'].shape[2] for item in batch])

    # Create empty tensors with the maximum dimensions
    input_batch = torch.zeros(len(batch), batch[0]['input'].shape[0], max_x, max_y)
    admin_mask_batch = torch.zeros(len(batch), max_x, max_y)
    y_batch = torch.zeros(len(batch))

    # Fill the tensors with the data from the batch
    for i, item in enumerate(batch):
        x_size, y_size = item['input'].shape[1], item['input'].shape[2]
        input_batch[i, :, :x_size, :y_size] = item['input']
        admin_mask_batch[i, :x_size, :y_size] = item['admin_mask']
        y_batch[i] = item['y']

    return {
        'input': input_batch,
        'admin_mask': admin_mask_batch,
        'y': y_batch,
        'img_coords': [item['img_coords'] for item in batch],
        'valid_coords': [item['valid_coords'] for item in batch],
        'season': torch.tensor([item['season'] for item in batch]),
        'source': torch.tensor([item['source'] for item in batch], dtype=torch.bool),
        'census_idx': torch.cat([item['census_idx'] for item in batch]),
    }


if __name__=="__main__":

    #test the dataset
    from torch.utils.data import DataLoader, ChainDataset, ConcatDataset

    input_defs = {'S1': True, 'S2': True, 'VIIRS': False, 'NIR': True}

    # Create the dataset for testing
    dataset = Population_Dataset_target("pri2017", mode="weaksup", patchsize=None, overlap=None, fourseasons=True, **input_defs) 
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True, drop_last=True)

    # Test the dataset
    for e in tqdm(range(10), leave=True):
        dataloader_iterator = iter(dataloader)
        for i in tqdm(range(5000)):
            sample = dataset[i%len(dataset)]
            print(i,sample['input'].shape)