

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

from utils.constants import pop_map_root_large, pop_map_root, pop_map_covariates, pop_map_covariates_large, config_path, pop_gbuildings_path, rawEE_map_root, skip_indices
from utils.constants import datalocations
from utils.plot import plot_2dmatrix
# from osgeo import gdal


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
                 max_samples=None, transform=None, sentinelbuildings=True, ascfill=False) -> None:
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
        self.sentinelbuildings = sentinelbuildings
        self.ascfill = ascfill

        # get the path to the data
        # region_root = os.path.join(pop_map_root_large, region)
        region_root = os.path.join(pop_map_root, region)

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
            # max_pix = 2e6
            # max_pix = 5e6
            max_pix = 1.25e6
            # max_pix = 1e16
            # max_pix = 1e16
            print("Kicking out ", (self.coarse_census["count"]>=max_pix).sum(), "samples with more than ", int(max_pix), " pixels")
            self.coarse_census = self.coarse_census[self.coarse_census["count"]<max_pix].reset_index(drop=True)

            # kicking out samples that are ugly shaped and difficualt to handle using the skip_indices
            self.coarse_census = self.coarse_census[~self.coarse_census["idx"].isin(skip_indices[region])].reset_index(drop=True)

            # redefine indexing
            if max_samples is not None:
                # self.coarse_census = self.coarse_census.sample(frac=1, random_state=1610)[:max_samples].reset_index(drop=True)
                self.coarse_census = self.coarse_census.sample(frac=1, random_state=1610)[-max_samples:].reset_index(drop=True)
            print("Using", len(self.coarse_census), "samples for weakly supervised training")

            # get the shape of the coarse regions
            with rasterio.open(self.file_paths["coarse"]["boundary"], "r") as src:
                self.cr_regions = src.read(1)
                self.cr_shape = src.shape


        elif self.mode=="test":
            # testdata specific preparation
            # get the shape and metadata of the images
            with rasterio.open(self.file_paths[list(self.file_paths.keys())[0]]["boundary"], "r") as src:
                self.img_shape = src.shape
                self._meta = src.meta.copy()
            self._meta.update(count=1, dtype='float32', nodata=None, compress='lzw')

            # get a list of indices of the possible patches
            self.patch_indices = self.get_patch_indices(patchsize, overlap)
        else:
            raise ValueError("Mode not recognized")

        # get the path to the data files
        covar_root = os.path.join(pop_map_covariates, region)
        # self.S1_file = os.path.join(covar_root,  os.path.join("S1", region +"_S1.tif"))
        S1spring_file = os.path.join(covar_root,  os.path.join("S1spring", region +"_S1spring.tif"))
        S1summer_file = os.path.join(covar_root,  os.path.join("S1summer", region +"_S1summer.tif"))
        S1autumn_file = os.path.join(covar_root,  os.path.join("S1autumn", region +"_S1autumn.tif"))
        S1winter_file = os.path.join(covar_root,  os.path.join("S1winter", region +"_S1winter.tif"))

        if ascfill:
            S1springAsc_file = os.path.join(covar_root,  os.path.join("S1springAsc", region +"_S1springAsc.tif"))
            S1summerAsc_file = os.path.join(covar_root,  os.path.join("S1summerAsc", region +"_S1summerAsc.tif"))
            S1autumnAsc_file = os.path.join(covar_root,  os.path.join("S1autumnAsc", region +"_S1autumnAsc.tif"))
            S1winterAsc_file = os.path.join(covar_root,  os.path.join("S1winterAsc", region +"_S1winterAsc.tif"))

            self.S1Asc_file = {0: S1springAsc_file, 1: S1summerAsc_file, 2: S1autumnAsc_file, 3: S1winterAsc_file}


        if not os.path.exists(S1spring_file):
            print("S1 file does not exist")
        
            spring_dir = os.path.join(rawEE_map_root, region, "S1spring")
            summer_dir = os.path.join(rawEE_map_root, region, "S1summer")
            autumn_dir = os.path.join(rawEE_map_root, region, "S1autumn")
            winter_dir = os.path.join(rawEE_map_root, region, "S1winter")

            if not os.path.exists(os.path.join(rawEE_map_root, region, "S1winter_out.vrt")):
                from osgeo import gdal
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1spring_out.vrt"), [ os.path.join(spring_dir, f) for f in os.listdir(spring_dir) if f.endswith(".tif")])
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1summer_out.vrt"), [ os.path.join(summer_dir, f) for f in os.listdir(summer_dir) if f.endswith(".tif")])
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1autumn_out.vrt"), [ os.path.join(autumn_dir, f) for f in os.listdir(autumn_dir) if f.endswith(".tif")])
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1winter_out.vrt"), [ os.path.join(winter_dir, f) for f in os.listdir(winter_dir) if f.endswith(".tif")])
                S1spring_file = os.path.join(rawEE_map_root, region, "S1spring_out.vrt")
                S1summer_file = os.path.join(rawEE_map_root, region, "S1summer_out.vrt")
                S1autumn_file = os.path.join(rawEE_map_root, region, "S1autumn_out.vrt")
                S1winter_file = os.path.join(rawEE_map_root, region, "S1winter_out.vrt")
                
            if ascfill:
                spring_dir = os.path.join(rawEE_map_root, region, "S1springAsc")
                summer_dir = os.path.join(rawEE_map_root, region, "S1summerAsc")
                autumn_dir = os.path.join(rawEE_map_root, region, "S1autumnAsc")
                winter_dir = os.path.join(rawEE_map_root, region, "S1winterAsc")

                if not os.path.exists(os.path.join(rawEE_map_root, region, "S1winterAsc_out.vrt")):
                    from osgeo import gdal
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1springAsc_out.vrt"), [ os.path.join(spring_dir, f) for f in os.listdir(spring_dir) if f.endswith(".tif")])
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1summerAsc_out.vrt"), [ os.path.join(summer_dir, f) for f in os.listdir(summer_dir) if f.endswith(".tif")])
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1autumnAsc_out.vrt"), [ os.path.join(autumn_dir, f) for f in os.listdir(autumn_dir) if f.endswith(".tif")])
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1winterAsc_out.vrt"), [ os.path.join(winter_dir, f) for f in os.listdir(winter_dir) if f.endswith(".tif")])
                    S1springAsc_file = os.path.join(rawEE_map_root, region, "S1springAsc_out.vrt")
                    S1summerAsc_file = os.path.join(rawEE_map_root, region, "S1summerAsc_out.vrt")
                    S1autumnAsc_file = os.path.join(rawEE_map_root, region, "S1autumnAsc_out.vrt")
                    S1winterAsc_file = os.path.join(rawEE_map_root, region, "S1winterAsc_out.vrt")

                self.S1Asc_file = {0: S1springAsc_file, 1: S1summerAsc_file, 2: S1autumnAsc_file, 3: S1winterAsc_file}


        self.S1_file = {0: S1spring_file, 1: S1summer_file, 2: S1autumn_file, 3: S1winter_file}
        
        if self.use2A:
            S2spring_file = os.path.join(covar_root,  os.path.join("S2Aspring", region +"_S2Aspring.tif"))
            S2summer_file = os.path.join(covar_root,  os.path.join("S2Asummer", region +"_S2Asummer.tif"))
            S2autumn_file = os.path.join(covar_root,  os.path.join("S2Aautumn", region +"_S2Aautumn.tif"))
            S2winter_file = os.path.join(covar_root,  os.path.join("S2Awinter", region +"_S2Awinter.tif"))
            
            # TODO: check if file exists
            # if not exists, we use the virtual rasters of the raw files
            # if exists, we use the preprocessed files

            if not os.path.exists(S2spring_file):
                print("Using virtual rasters for S2")
                
                spring_dir = os.path.join(rawEE_map_root, region, "S2Aspring")
                summer_dir = os.path.join(rawEE_map_root, region, "S2Asummer")
                autumn_dir = os.path.join(rawEE_map_root, region, "S2Aautumn")
                winter_dir = os.path.join(rawEE_map_root, region, "S2Awinter")

                if not os.path.exists(os.path.join(rawEE_map_root, region, "S2Awinter_out.vrt")):
                    from osgeo import gdal
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S2Aspring_out.vrt"), [ os.path.join(spring_dir, f) for f in os.listdir(spring_dir) if f.endswith(".tif")])
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S2Asummer_out.vrt"), [ os.path.join(summer_dir, f) for f in os.listdir(summer_dir) if f.endswith(".tif")])
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S2Aautumn_out.vrt"), [ os.path.join(autumn_dir, f) for f in os.listdir(autumn_dir) if f.endswith(".tif")])
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S2Awinter_out.vrt"), [ os.path.join(winter_dir, f) for f in os.listdir(winter_dir) if f.endswith(".tif")])
                    S2spring_file = os.path.join(rawEE_map_root, region, "S2Aspring_out.vrt")
                    S2summer_file = os.path.join(rawEE_map_root, region, "S2Asummer_out.vrt")
                    S2autumn_file = os.path.join(rawEE_map_root, region, "S2Aautumn_out.vrt")
                    S2winter_file = os.path.join(rawEE_map_root, region, "S2Awinter_out.vrt")

        else:
            S2spring_file = os.path.join(covar_root,  os.path.join("S21Cspring", region +"_S21Cspring.tif"))
            S2summer_file = os.path.join(covar_root,  os.path.join("S21Csummer", region +"_S21Csummer.tif"))
            S2autumn_file = os.path.join(covar_root,  os.path.join("S21Cautumn", region +"_S21Cautumn.tif"))
            S2winter_file = os.path.join(covar_root,  os.path.join("S21Cwinter", region +"_S21Cwinter.tif"))

        self.S2_file = {0: S2spring_file, 1: S2summer_file, 2: S2autumn_file, 3: S2winter_file}
        self.season_dict = {0: "spring", 1: "summer", 2: "autumn", 3: "winter"}
        self.inv_season_dict = {v: k for k, v in self.season_dict.items()}
        self.VIIRS_file = os.path.join(covar_root,  os.path.join("viirs", region +"_viirs.tif"))

        # load the google buildings
        if self.sentinelbuildings:
            # load sentinel buildings
            self.sbuildings_segmentation_file = os.path.join(pop_map_root, region, "buildingsDDA2_44C.tif")
            self.gbuildings_segmentation_file = ''
            # self.sbuildings_segmentation_file = os.path.join(pop_map_root, region, "buildingsDDA128_4096_nodisc.tif")
            self.sbuildings = True
            self.gbuildings = False
        else:
            self.sbuildings_segmentation_file = ''
            self.gbuildings_segmentation_file = os.path.join(pop_gbuildings_path, region, "Gbuildings_" + region + "_segmentation.tif")
            self.gbuildings_counts_file = os.path.join(pop_gbuildings_path, region, "Gbuildings_" + region + "_counts.tif")
            self.gbuildings = True
        self.gbuildings = False

        # normalize the dataset (do not use, this does not make sense for variable regions sizes like here)
        self.y_stats = load_json(os.path.join(config_path, 'dataset_stats', 'label_stats.json'))
        self.y_stats['max'] = float(self.y_stats['max'])
        self.y_stats['min'] = float(self.y_stats['min'])

    # delete the dataset
    def __del__(self):
        pass
        for file in self.S1_file.values():
            if isinstance(file, gdal.Dataset):
                file = None
        for file in self.S2_file.values():
            if isinstance(file, gdal.Dataset):
                file = None

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
        # xmin, xmax, ymin, ymax = tuple(map(int, census_sample["bbox"].strip('()').split(',')))
        xmin, xmax, ymin, ymax = tuple(map(int, census_sample["bbox"].strip('()').strip('[]').split(',')))

        # get the season for the S2 data
        season = random.choice(['spring', 'autumn', 'winter', 'summer']) if self.fourseasons else "spring"
        # season = random.choice(['spring', 'autumn', 'winter', 'summer']) if self.fourseasons else "autumn"

        # get the data
        ad_over = 32
        indata, auxdata, w = self.generate_raw_data(xmin, ymin, self.inv_season_dict[season], patchsize=(xmax-xmin, ymax-ymin), overlap=0, admin_overlap=ad_over)

        if "S2" in indata:
            if np.any(np.isnan(indata["S2"])): 
                indata["S2"] = self.interpolate_nan(indata["S2"]) 
            
        if "S1" in indata:
            if np.any(np.isnan(indata["S1"])):
                if torch.isnan(torch.tensor(indata["S1"])).sum() / torch.numel(torch.tensor(indata["S1"])) < 0.05 and not self.ascfill:
                    indata["S1"] = self.interpolate_nan(indata["S1"])
                else:
                    # generate another datapatch, but with the ascending orbit
                    indataAsc, _, _ = self.generate_raw_data(xmin, ymin, self.inv_season_dict[season], patchsize=(xmax-xmin, ymax-ymin), overlap=0, admin_overlap=ad_over, descending=False)
                    indata["S1"] = indataAsc["S1"]
                    if torch.isnan(torch.tensor(indata["S1"])).sum() / torch.numel(torch.tensor(indata["S1"])) < 0.05:
                        indata["S1"] = self.interpolate_nan(indata["S1"])
                    else:
                        print("S1 contains too many NaNs, skipping")
                        raise Exception("No data here!")
                    
        # get admin_mask
        admin_mask = torch.from_numpy(self.cr_regions[w[0][0]:w[0][1], w[1][0]:w[1][1]])

        # To Torch
        indata = {key:torch.from_numpy(np.asarray(val, dtype=np.float32)).type(torch.FloatTensor) for key,val in indata.items()}

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
        """
        Description:
            Get the item at the given index, the item is a patch
        Input:
            index: index of the item
        Output:
            item: dictionary containing the input, the mask, the coordinates of the patch, the season and the season string
        """

        # get the indices of the patch
        x,y,season = self.patch_indices[index]

        # get the data
        indata, mask, window = self.generate_raw_data(x,y,season.item())

        if "S2" in indata:
            if np.any(np.isnan(indata["S2"])): 
                indata["S2"] = self.interpolate_nan(indata["S2"]) 
            
        if "S1" in indata:
            # if np.any(np.isnan(indata["S1"])):
            #     indata["S1"] = self.interpolate_nan(indata["S1"]) 
            if np.any(np.isnan(indata["S1"])):
                if torch.isnan(torch.tensor(indata["S1"])).sum() / torch.numel(torch.tensor(indata["S1"])) < 0.05 and not self.ascfill:
                    indata["S1"] = self.interpolate_nan(indata["S1"])
                else:
                    # generate another datapatch, but with the ascending orbit
                    indataAsc, mask, window = self.generate_raw_data(x,y,season.item(), descending=False)
                    # indataAsc, mask, window = self.generate_raw_data(x,y, season=2, descending=False)
                    # indataAsc, _, _ = self.generate_raw_data(xmin, ymin, self.inv_season_dict[season], patchsize=(xmax-xmin, ymax-ymin), overlap=0, admin_overlap=ad_over, descending=False)
                    indata["S1"] = indataAsc["S1"]
                    if np.any(np.isnan(indata["S1"])):
                        if torch.isnan(torch.tensor(indata["S1"])).sum() / torch.numel(torch.tensor(indata["S1"])) < 0.05:
                            indata["S1"] = self.interpolate_nan(indata["S1"])
                        else:
                            print("S1 contains too many NaNs, skipping")
                            raise Exception("No data here!")
        # To Torch
        indata = {key:torch.from_numpy(np.asarray(val, dtype=np.float32)).type(torch.FloatTensor) for key,val in indata.items()} 
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        # get valid coordinates of the patch
        xmin = x+self.overlap
        xmax = x+self.patchsize-self.overlap
        ymin = y+self.overlap
        ymax = y+self.patchsize-self.overlap

        # return dictionary
        return {
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

        if (~nan_mask).sum()< 4:
            print("all nan detected")
            return np.zeros_like(input_array)
        
        # interpolate the missing values
        interpolated_values = griddata(np.vstack(known_points).T, values, np.vstack(missing_points).T, method='nearest')

        # fillin the missing ones
        input_array[missing_points] = interpolated_values

        return input_array


    def generate_raw_data(self, x, y, season, patchsize=None, overlap=None, admin_overlap=0, descending=True):
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

        # get the window of the patch for administative region case
        if admin_overlap>0:
            new_x = max(x-admin_overlap,0)
            new_y = max(y-admin_overlap,0)
            x_stop = min(x+patchsize_x+admin_overlap, self.cr_shape[0])
            y_stop = min(y+patchsize_y+admin_overlap, self.cr_shape[1])
            window = ((new_x,x_stop),(new_y,y_stop))

        else:
            window = ((x,x+patchsize_x),(y,y+patchsize_y))

        indata = {}
        mask = np.zeros((patchsize_x, patchsize_y), dtype=bool)
        mask[overlap:patchsize_x-overlap, overlap:patchsize_y-overlap] = True

        # for debugging
        fake = False

        # get the input data
        if self.S2:
            S2_file = self.S2_file[season]
            if self.NIR:
                if fake:
                    indata["S2"] = np.random.randint(0, 10000, size=(4,patchsize_x,patchsize_y))

                # elif isinstance(S2_file, gdal.Dataset):
                #     indata["S2"] = self.read_gdal_file(S2_file, (3,2,1,4), window=window)
                else:
                    with rasterio.open(S2_file, "r") as src:
                        indata["S2"] = src.read((3,2,1,4), window=window).astype(np.float32) 
            else:
                if fake:
                    indata["S2"] = np.random.randint(0, 10000, size=(3,patchsize_x,patchsize_y))
                # elif isinstance(S2_file, gdal.Dataset):
                #     indata["S2"] = self.read_gdal_file(S2_file, (3,2,1), window=window)
                else:
                    with rasterio.open(S2_file, "r") as src:
                        indata["S2"] = src.read((3,2,1), window=window).astype(np.float32) 
            # mask = mask & (indata["S2"].sum(axis=0) != 0)
        if self.S1:
            S1_file = self.S1_file[season] if descending else self.S1Asc_file[season]
            if fake:
                indata["S1"] = np.random.randint(0, 10000, size=(2,patchsize_x,patchsize_y))
            # elif isinstance(S1_file, gdal.Dataset):
            #     indata["S1"] = self.read_gdal_file(S1_file, (1,2), window=window)
            else:
                with rasterio.open(S1_file, "r") as src:
                    indata["S1"] = src.read((1,2), window=window).astype(np.float32) 
            # mask = mask & (indata["S1"].sum(axis=0) != 0)
        if self.VIIRS:
            if fake:
                indata["VIIRS"] = np.random.randint(0, 10000, size=(1,patchsize_x,patchsize_y))
            # elif isinstance(self.VIIRS_file, gdal.Dataset):
            #     indata["VIIRS"] = self.read_gdal_file(self.VIIRS_file, (1,), window=window)
            else:
                with rasterio.open(self.VIIRS_file, "r") as src:
                    indata["VIIRS"] = src.read(1, window=window).astype(np.float32)  

        if self.gbuildings or self.sentinelbuildings:
            if fake:
                indata["building_segmentation"] = np.random.randint(0, 1, size=(1,patchsize_x,patchsize_y))
                indata["building_counts"] = np.random.randint(0, 2, size=(1,patchsize_x,patchsize_y))
            else:
                if self.sentinelbuildings and os.path.exists(self.sbuildings_segmentation_file):
                    with rasterio.open(self.sbuildings_segmentation_file, "r") as src:
                        indata["building_counts"] = src.read(1, window=window)[np.newaxis].astype(np.float32)/255

                elif os.path.exists(self.gbuildings_segmentation_file): 
                    with rasterio.open(self.gbuildings_segmentation_file, "r") as src:
                        indata["building_segmentation"] = src.read(1, window=window)[np.newaxis].astype(np.float32)
                    with rasterio.open(self.gbuildings_counts_file, "r") as src:
                        indata["building_counts"] = src.read(1, window=window)[np.newaxis].astype(np.float32) 

        # # load administrative mask
        # admin_mask = self.cr_regions[x:x+patchsize_x, y:y+patchsize_y]==idx

        return indata, mask, window
    

    def read_gdal_file(self, file, bands, window=None):
        """
        Reads a gdal file and returns the data
        inputs:
            :param file: the gdal file
            :param bands: the bands to read, 1-based indexing like in gdal (e.g. (3,2,1) for RGB)
            :param window: the window to read
        outputs:
            :return: the data
        """
        with rasterio.open(file.GetDescription(), 'r') as raster_vrt:
            bands_out = raster_vrt.read(bands, window=window).astype(np.float32) 
        return bands_out
    

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

        # raise NotImplementedError
        with rasterio.open(boundary_file, "r") as src:
            boundary = src.read(1)
        boundary = torch.from_numpy(boundary)

        # read the census file
        census = pd.read_csv(census_file)

        if gpu_mode:
            # pred = pred.to(torch.float32).cuda()
            pred = pred.cuda()
            boundary = boundary.cuda()
            # census_pred = torch.zeros(len(census), dtype=torch.float32).cuda()
            census_pred = -torch.ones(census["idx"].max()+1, dtype=torch.float32).cuda()

            # iterate over census regions and get totals
            # for i, (cidx,bbox) in tqdm(enumerate(zip(census["idx"], census["bbox"])), total=len(census)):
            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
            # for i, (cidx,bbox) in tqdm(enumerate(zip(census["idx"], census["bbox"]))):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                census_pred[cidx] = pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx].to(torch.float32).sum()
                # census["count"] = (boundary[xmin:xmax, ymin:ymax]==cidx).to(torch.float32).sum()

        else:
            pred = pred.to(torch.float32)
            census_pred = torch.zeros(len(census), dtype=torch.float32)

            # iterate over census regions and get totals
            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                # xmin, xmax, ymin, ymax = tuple(map(int, tuple(map(int, bbox.strip('()').strip('[]').split(','))).split(',')))
                census_pred[cidx] = pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx].sum()
                # census_pred[i] = pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx].sum()
        
        valid_census = census_pred>-1
        census_pred = census_pred[valid_census]
        # census = census[valid_census]

        # # produce density map
        # densities = torch.zeros_like(pred)
        # pred_densities_census = census_pred.cpu() / census["count"]
        # for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
        #     xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
        #     densities[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] = pred_densities_census[i]

        # # total map
        # totals = torch.zeros_like(pred, dtype=torch.float32)
        # totals_pred_census = census_pred.cpu().to(torch.float32)
        # for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
        #     xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
        #     totals[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] = totals_pred_census[i]

        # # produce density map for the ground truth
        # densities_gt = torch.zeros_like(pred)
        # gt_densities_census = torch.tensor(census["POP20"]) / census["count"]
        # for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
        #     xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
        #     densities_gt[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] = gt_densities_census[i]

        # # total map
        # totals_gt = torch.zeros_like(pred, dtype=torch.float32)
        # totals_gt_census = torch.tensor(census["POP20"]).to(torch.float32)
        # for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
        #     xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
        #     totals_gt[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] = totals_gt_census[i]


        del boundary, pred
        torch.cuda.empty_cache()

        return census_pred, torch.tensor(census["POP20"])
    
    def adjust_map_to_census(self, pred, gpu_mode=True):
        """
        Adjust the predicted map to the census regions via dasymmetric mapping strategy
        inputs:
            :param pred: predicted map
            :param census: census data
        outputs:
            :return: adjusted map
        """
        boundary_file = self.file_paths["coarse"]["boundary"]
        census_file = self.file_paths["coarse"]["census"]

        # raise NotImplementedError
        with rasterio.open(boundary_file, "r") as src:
            boundary = src.read(1)
        boundary = torch.from_numpy(boundary)

        # read the census file
        census = pd.read_csv(census_file)

        # iterate over census regions and adjust the totals
        for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
            xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
            pred_census_count = pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx].to(torch.float32).sum()
            adj_scale = census["POP20"][i] / pred_census_count
            pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] *= adj_scale

        return pred
    

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

        try:
            with rasterio.open(output_file, "w", CHECK_DISK_FREE_SPACE="NO", **self._meta) as dest:
                dest.write(preds,1)
        except:
            print("Error saving predictions to file, continuing...")
            pass



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
    max_x = max([item['S1'].shape[1] for item in batch])
    max_y = max([item['S1'].shape[2] for item in batch])

    # Create empty tensors with the maximum dimensions
    input_batch_S1 = torch.zeros(len(batch), batch[0]['S1'].shape[0], max_x, max_y)
    input_batch_S2 = torch.zeros(len(batch), batch[0]['S2'].shape[0], max_x, max_y)

    building_segmentation = torch.zeros(len(batch), 1, max_x, max_y),
    building_counts = torch.zeros(len(batch), 1, max_x, max_y)
    
    use_building_segmentation = False

    admin_mask_batch = torch.zeros(len(batch), max_x, max_y)
    y_batch = torch.zeros(len(batch))

    # Fill the tensors with the data from the batch
    for i, item in enumerate(batch):
        # x_size, y_size = item['input'].shape[1], item['input'].shape[2]
        # input_batch[i, :, :x_size, :y_size] = item['input']
        x_size, y_size = item['S1'].shape[1], item['S1'].shape[2]
        input_batch_S1[i, :, :x_size, :y_size] = item['S1']
        input_batch_S2[i, :, :x_size, :y_size] = item['S2']

        admin_mask_batch[i, :x_size, :y_size] = item['admin_mask']
        y_batch[i] = item['y']

        if "building_segmentation" in item:
            building_segmentation[i, :, :x_size, :y_size] = item['building_segmentation']
            use_building_segmentation = True
        if "building_counts" in item:
            building_counts[i, :, :x_size, :y_size] = item['building_counts']
            use_building_counts = True

    out_dict = {
        'S1': input_batch_S1,
        'S2': input_batch_S2,
        'admin_mask': admin_mask_batch,
        'y': y_batch,
        'img_coords': [item['img_coords'] for item in batch],
        'valid_coords': [item['valid_coords'] for item in batch],
        'season': torch.tensor([item['season'] for item in batch]),
        'source': torch.tensor([item['source'] for item in batch], dtype=torch.bool),
        'census_idx': torch.cat([item['census_idx'] for item in batch]),
    }
    if use_building_segmentation:
        out_dict["building_segmentation"] = building_segmentation
    if use_building_counts:
        out_dict["building_counts"] = building_counts

    return out_dict


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