import json
import os
from os.path import isfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
import torch
from joblib import load
from rasterio.enums import Resampling
from torch.utils.data import Dataset
from utils.constants import img_rows, img_cols, osm_features, num_classes, config_path
from utils.plot import plot_2dmatrix
import random
from scipy.interpolate import griddata

from torch.utils.data import DataLoader

from tqdm import tqdm

# mm_scaler = load(os.path.join(config_path, 'dataset_stats', 'mm_scaler.joblib'))


def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


# import torchdatasets as td
class PopulationDataset_Reg(Dataset):
    """
    Population Dataset for Regression Task
    """
    def __init__(self, list_IDs, labels, f_names_unlab=[], dim=(img_rows, img_cols), transform=None, test=False, mode=None, satmode=False, in_memory=False,
                 S1=False, S2=True, VIIRS=True, NIR=False, random_season=False): 

        self.dim = dim
        self.labels = labels
        self.list_IDs = list_IDs
        self.f_names_unlab = f_names_unlab
        self.n_classes = num_classes
        self.transform = transform
        self.test = test
        self.mode = mode 
        self.in_memory = in_memory
        self.move_to_memory = False
        self.S1 = S1
        self.S2 = S2
        self.NIR =NIR
        self.VIIRS = VIIRS
        self.random_season = random_season                 
        self.satmode = satmode

        self.use2A = True

        self.all_ids = list_IDs + f_names_unlab
        self.labeled_indices = [i for i,name in enumerate(self.all_ids) if name in list_IDs]
        self.unlabeled_indices = [i for i,name in enumerate(self.all_ids) if name in f_names_unlab]
 
        self.y_stats = load_json(os.path.join(config_path, 'dataset_stats', 'label_stats.json'))
        self.y_stats['max'] = float(self.y_stats['max'])
        self.y_stats['min'] = float(self.y_stats['min'])

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


    def __getitem__(self, idx):
        # if self.in_memory and not self.move_to_memory:
        #     sample = self.all_samples[self.all_ids[idx]] 
        #     if self.transform:
        #         sample["input"] = self.transform(sample["input"])
        #     return sample

        # query the data ID
        ID_temp = self.all_ids[idx]

        # Generate data
        indata, auxdata = self.generate_raw_data(ID_temp)

        # ID = ID_temp.split(os.sep)[-1].split('_sen2')[0]
        # ID = ID_temp.split(os.sep)[-1].split('_S2')[0]
        ID = ID_temp.split(os.sep)[-1].split('_S2A')[0]

        # Generate labels if posssible
        if idx<len(self.labels):
            y = self.labels[idx]
            y_norm = self.normalize_reg_labels(y) 
            y = torch.from_numpy(np.asarray(y)).type(torch.FloatTensor)
            y_norm = torch.from_numpy(np.asarray(y_norm)).type(torch.FloatTensor)
            source = True
        else:
            y = torch.tensor(-9999).type(torch.FloatTensor)
            y_norm = torch.tensor(-9999).type(torch.FloatTensor)
            source = False

        # To torch
        indata = {key:torch.from_numpy(np.asarray(val, dtype=np.float32)).type(torch.FloatTensor) for key,val in indata.items()} 
        auxdata = {key:torch.from_numpy(np.asarray(val, dtype=np.float32)).type(torch.FloatTensor) for key,val in auxdata.items()} 

        if self.move_to_memory:
            return {**indata, **auxdata, 'y': y, 'y_norm': y_norm, 'identifier': ID, 'source': source}
        
        # Modality-wise transformations
        # if self.transform:
        #     if "S2" in self.transform and "S2" in indata:
        #         indata["S2"] = self.transform["S2"](indata["S2"])
        #     if "S1" in self.transform and "S1" in indata:
        #         indata["S1"] = self.transform["S1"](indata["S1"])

        # Normalizations
        # indata = self.normalize_indata(indata)

        # merge inputs
        # X = torch.concatenate([indata[key] for key in ["S2", "S1", "VIIRS"] if key in indata], dim=0)

        # # General transformations
        # if self.transform:
        #     X = self.transform["general"](X) if "general" in self.transform.keys() else X
        
        # Collect all variables
        sample = {
            # 'input': X,
            **auxdata, 'y': y, 'y_norm': y_norm, 'identifier': ID, 'source': source,
            **indata}
        return sample


    def __len__(self):
        if self.mode=="train":
            return len(self.labels)
        else:
            return len(self.labels)
    

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
    
    def generate_raw_data(self, ID_temp, trials=0):
        """
        
        """
        # Initialization
        # preparing the batch from other datasets
        
        # basetype = "sen2spring"
        # basetype = "S2spring"
        basetype = "S2Aspring"
        # base = "S2spring"
        if self.random_season:
            # ID_sen2 = ID_temp.replace(basetype, "sen2{}".format(random.choice(['spring', 'autumn', 'winter', 'summer'])))
            # ID_sen2 = ID_temp.replace(basetype, "S2{}".format(random.choice(['spring', 'autumn', 'winter', 'summer'])))
            if self.use2A:
                ID_sen2 = ID_temp.replace(basetype, "S2A{}".format(random.choice(['spring', 'autumn', 'winter', 'summer'])))
            else:
                ID_sen2 = ID_temp.replace(basetype, "S21C{}".format(random.choice(['spring', 'autumn', 'winter', 'summer'])))
            ID_sen1 = ID_temp.replace(basetype, "S1{}".format(random.choice(['spring', 'autumn', 'winter', 'summer'])))
        else:
            if self.use2A:
                ID_sen2 = ID_temp.replace(basetype, "S2Aspring") # for testing just use the spring images
            else:
                ID_sen2 = ID_temp.replace(basetype, "S21Cspring") # for testing just use the spring images
            ID_sen1 = ID_temp.replace(basetype, "S1spring") # for testing just use the spring images
                
        # ID_sen1 = ID_temp.replace(basetype, 'S1')
        ID_viirs = ID_temp.replace(basetype, 'viirs')
        # ID_osm = ID_temp.replace(basetype, 'osm_features').replace('tif', 'csv')
        # ID_lu = ID_temp.replace(basetype, 'lu')
        ID_Pop = ID_temp.replace(basetype, 'Pop')
        ID_PopNN = ID_temp.replace(basetype, 'PopNN')
        ID_msb = ID_temp.replace(basetype, 'msb')
        
        # get the input data
        fake = False
        indata = {}
        if self.S2: 
            if self.NIR:
                if fake:
                    indata["S2"] = np.random.randint(0, 10000, size=(4,img_rows, img_cols))
                else:
                    indata["S2"] = np.zeros((4,img_rows, img_cols))
                    with rasterio.open(ID_sen2, "r") as src:
                        # indata["S2"] = src.read((4,3,2,8))
                        indata["S2"][:] = src.read((3,2,1,4)).astype(np.float32) 
            else:
                if fake:
                    indata["S2"] = np.random.randint(0, 10000, size=(3,img_rows, img_cols))
                else:
                    indata["S2"] = np.zeros((3,img_rows, img_cols))
                    with rasterio.open(ID_sen2, "r") as src:
                        # indata["S2"] = src.read((4,3,2))
                        indata["S2"][:] = src.read((3,2,1)).astype(np.float32) 

            # if torch.isnan(torch.tensor(indata["S2"])).any():
            # indata["S2"] = indata["S2"].astype(np.float32)
            if np.isnan(indata["S2"]).any():
                if torch.isnan(torch.tensor(indata["S2"])).sum() / torch.numel(torch.tensor(indata["S2"])) < 0.2:
                    indata["S2"] = self.interpolate_nan(indata["S2"])
                elif trials < 16:
                    return self.generate_raw_data_new_sample(ID_temp, trials=trials)
                    # print("Too many NaNs in S2 image, recursive procedure. Trial:", trials) 
                    # plot_2dmatrix(b["S2"]/3500)
                else:
                    print("Too many NaNs in S2 image, skipping sample")
                    print("Suspect:", ID_temp)
                    raise Exception("Too many NaNs in S2 image, breaking")
                
            assert indata["S2"].shape[1] == img_rows
            assert indata["S2"].shape[2] == img_cols

        if self.S1:
            if fake:
                indata["S1"] = np.random.randint(0, 10000, size=(2,img_rows, img_cols))
            else:
                indata["S1"] = np.zeros((2,img_rows, img_cols))
                with rasterio.open(ID_sen1, "r") as src:
                    indata["S1"][:] = src.read((1,2)).astype(np.float32)
                    # src.close()
                    # del src
                # indata["S1"] = np.random.randint(0, 10000, size=(2,img_rows, img_cols))

            assert indata["S1"].shape[1] == img_rows
            assert indata["S1"].shape[2] == img_cols

        if self.VIIRS:
            if fake:
                indata["VIIRS"] = np.random.randint(0, 10000, size=(1,img_rows, img_cols))
            else:
                with rasterio.open(ID_viirs, "r") as src:
                    indata["VIIRS"] = src.read(1).astype(np.float32)

        auxdata = {}
        # if finegrained cencus is available
        if self.mode=="train": 
            auxdata["Pop_X"]  = np.zeros((0,0))
            auxdata["PopNN_X"] = np.zeros((0,0))
            auxdata["pop_avail"] = np.array([0])
        elif isfile(ID_Pop):
            with rasterio.open(ID_Pop, "r") as src:
                auxdata["Pop_X"] = src.read(1)
            with rasterio.open(ID_PopNN, "r") as src:
                auxdata["PopNN_X"] = src.read(1)
            auxdata["pop_avail"] = np.array([1])
        else:
            auxdata["Pop_X"] = np.zeros((10,10))
            auxdata["PopNN_X"] = np.zeros((100,100))
            auxdata["pop_avail"] = np.array([0])

        use_msb = False
        if isfile(ID_msb) and use_msb:
            msb_avail = np.array([1])
        else:
            msb = np.zeros((100,100))
            msb_avail = np.array([0])
        
        return indata, auxdata
    
    def generate_raw_data_new_sample(self, ID_temp, trials):

        if trials > 4:
            print("Too many trials, breaking")
            raise Exception("Enough trials")
        
        for season in ["spring", "summer", "autumn", "winter"]:
            # ID_temp = ID_temp.replace("sen2spring", "sen2{}".format(season))
            # ID_temp = ID_temp.replace("sen2spring", "sen2{}".format(season))
            if self.use2A:
                # ID_temp = ID_temp.replace("S2Aspring", "S2A{}".format(season))
                ID_temp = ID_temp.replace("spring", "{}".format(season))
            else:
                # ID_temp = ID_temp.replace("S2Aspring", "S21C{}".format(season))
                ID_temp = ID_temp.replace("spring", "{}".format(season))
            # ID_temp = ID_temp.replace("S2Aspring", "sen2{}".format(season))
            return self.generate_raw_data(ID_temp, trials=trials+1)


    def normalize_indata(self,indata):

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
        y_max = self.y_stats['max']
        y_min = self.y_stats['min']
        y_scaled = (y - y_min) / (y_max - y_min)
        return y_scaled

    def denormalize_reg_labels(self, y_scaled):  
        y_max = self.y_stats['max']
        y_min = self.y_stats['min']
        y = y_scaled * (y_max - y_min) + y_min
        return y