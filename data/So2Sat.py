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
from utils.utils import plot_2dmatrix

from torch.utils.data import DataLoader

from tqdm import tqdm

mm_scaler = load(os.path.join(config_path, 'dataset_stats', 'mm_scaler.joblib'))


def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


# import torchdatasets as td
class PopulationDataset_Reg(Dataset):
    """
    Population Dataset for Standard Regression Task
    """
    def __init__(self, list_IDs, labels, dim=(img_rows, img_cols), transform=None, test=False, mode=None, satmode=False, in_memory=False,
                 S1=False, S2=True, VIIRS=True): 

        self.dim = dim
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = num_classes
        self.transform = transform
        self.test = test
        self.mode = mode 
        self.in_memory = in_memory
        self.move_to_memory = False
        self.S1 = S1
        self.S2 = S2
        self.VIIRS = VIIRS
                                                    
        self.satmode = satmode
        self.y_stats = load_json(os.path.join(config_path, 'dataset_stats', 'label_stats.json'))
        self.y_stats['max'] = float(self.y_stats['max'])
        self.y_stats['min'] = float(self.y_stats['min'])
        self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats.json'))
        for mkey in self.dataset_stats.keys():
            if isinstance(self.dataset_stats[mkey], dict):
                for key,val in self.dataset_stats[mkey].items():
                    self.dataset_stats[mkey][key] = np.array(val)
            else:
                self.dataset_stats[mkey] = np.array(val)

        # Memory Mode
        self.all_samples = {}
        if in_memory:
            print("Loading to memory for Dataset: ", mode)
            self.move_to_memory = True              
            for idx in tqdm(range(len(self))):
                self.all_samples[self.list_IDs[idx]] = self[idx]
            self.move_to_memory = False 
                

    def __getitem__(self, idx):
        if self.in_memory and not self.move_to_memory:
            sample = self.all_samples[self.list_IDs[idx]]

            if self.transform:
                sample["input"] = self.transform(sample["input"])

            return sample

        ID_temp = self.list_IDs[idx]
        # Generate data
        if self.satmode:
            X, Pop_X, PopNN_X, pop_avail, msb, msb_avail = self.data_generation(ID_temp)
        else:
            X, osm = self.data_generation(ID_temp)

        ID = ID_temp.split(os.sep)[-1].split('_sen2')[0]
        y = self.labels[idx]
        y_norm = self.normalize_reg_labels(y) 

        # To torch
        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(np.asarray(y)).type(torch.FloatTensor)
        y_norm = torch.from_numpy(np.asarray(y_norm)).type(torch.FloatTensor)
        Pop_X = torch.from_numpy(np.asarray(Pop_X)).type(torch.FloatTensor)
        PopNN_X = torch.from_numpy(np.asarray(PopNN_X)).type(torch.FloatTensor)
        pop_avail = torch.from_numpy(np.asarray(pop_avail)).type(torch.FloatTensor)
        msb = torch.from_numpy(np.asarray(msb)).type(torch.FloatTensor) 
        msb_avail = torch.from_numpy(np.asarray(msb_avail)).type(torch.FloatTensor)

        if self.move_to_memory:
            return {'input': X, 'y': y, 'y_norm': y_norm, 
                      'Pop_X': Pop_X, 'PopNN_X': PopNN_X, 'pop_avail': pop_avail,
                      'builtupmap': msb,  'msb_avail': msb_avail,
                      'identifier': ID}

        if self.transform:
            X = self.transform(X)

        if self.satmode:
            sample = {'input': X, 'y': y, 'y_norm': y_norm, 
                      'Pop_X': Pop_X, 'PopNN_X': PopNN_X, 'pop_avail': pop_avail,
                      'builtupmap': msb,  'msb_avail': msb_avail,
                      'identifier': ID}
        else:
            sample = {'input': X, 'y': y, 'y_norm': y_norm, 'osm': osm, 'identifier': ID}

        return sample


    def __len__(self):
        return len(self.labels)
    

    def data_generation(self, ID_temp):
        # Initialization
        if ID_temp is None:
            sen2spr_X = np.zeros((3, *self.dim))
            viirs_X = np.zeros((1, *self.dim))
            osm_X = np.zeros((osm_features, 1))
            lcz_X = np.zeros((1, *self.dim))
            lu_X = np.zeros((4, *self.dim))
            dem_X = np.zeros((1, *self.dim))
        else:
            if self.satmode: 
                # preparing the batch from other datasets
                ID_spring = ID_temp  # batch from sen2 autumn
                ID_sen1 = ID_temp.replace('sen2spring', 'S1')
                ID_viirs = ID_temp.replace('sen2spring', 'viirs')
                ID_osm = ID_temp.replace('sen2spring', 'osm_features').replace('tif', 'csv')
                ID_lu = ID_temp.replace('sen2spring', 'lu')
                ID_Pop = ID_temp.replace('sen2spring', 'Pop')
                ID_PopNN = ID_temp.replace('sen2spring', 'PopNN')

                # dem_X = generate_data(dem_X, ID_dem, channels=1, data='dem')
                # TODO: check if MS buildings exist, and load them
                # TODO: load S1 imagery
                
                data = []
                if self.S2:
                    data.append(self.generate_data(ID_spring, channels=3, data='sen2spring'))
                if self.S1:
                    data.append(self.generate_data(ID_sen1, channels=2 , data='sen1'))
                if self.VIIRS:
                    data.append(self.generate_data(ID_viirs, channels=1, data='viirs'))
                # osm_X = self.generate_osm_data(osm_X, ID_osm, mm_scaler, channels=1)
                lu_X = self.generate_data(ID_lu, channels=4, data='lu') 

                # if finegrained cencus is available
                if isfile(ID_Pop):
                    Pop_X = self.generate_data(ID_Pop, channels=1, data='Pop') 
                    PopNN_X = self.generate_data(ID_PopNN, channels=1, data='PopNN')
                    pop_avail = np.array([1])
                elif self.mode=="train":
                    Pop_X = np.zeros((0,0))
                    PopNN_X = np.zeros((0,0))
                    pop_avail = np.array([0])
                else:
                    Pop_X = np.zeros((10,10))
                    PopNN_X = np.zeros((100,100))
                    pop_avail = np.array([0])

                msb_avail = np.array([0])

                # return np.concatenate((sen2spr_X, viirs_X), axis=0), np.array([0]) , np.argmax(lu_X,0)>1.5
                return np.concatenate(data, axis=0), Pop_X, PopNN_X, pop_avail, np.argmax(lu_X,0)>1.5, msb_avail
            else:
                sen2spr_X = np.empty((*self.dim, 3))
                viirs_X = np.empty((*self.dim, 1))
                osm_X = np.empty((osm_features, 1))
                lcz_X = np.empty((*self.dim, 1))
                lu_X = np.empty((*self.dim, 4))
                dem_X = np.empty((*self.dim, 1))

                # preparing the batch from other datasets
                ID_spring = ID_temp  # batch from sen2 autumn
                ID_viirs = ID_temp.replace('sen2spring', 'viirs')
                ID_osm = ID_temp.replace('sen2spring', 'osm_features').replace('tif', 'csv')
                ID_lcz = ID_temp.replace('sen2spring', 'lcz')
                ID_lu = ID_temp.replace('sen2spring', 'lu')
                ID_dem = ID_temp.replace('Part1', 'Part2').replace('sen2spring', 'dem') 

                sen2spr_X = self.generate_data(sen2spr_X, ID_spring, channels=3, data='sen2spring')
                viirs_X = self.generate_data(viirs_X, ID_viirs, channels=1, data='viirs')
                osm_X = self.generate_osm_data(osm_X, ID_osm, mm_scaler, channels=1)
                lcz_X = self.generate_data(lcz_X, ID_lcz, channels=1, data='lcz')
                lu_X = self.generate_data(lu_X, ID_lu, channels=4, data='lu')
                dem_X = self.generate_data(dem_X, ID_dem, channels=1, data='dem')

                return np.concatenate((sen2spr_X, viirs_X, lcz_X, lu_X, dem_X), axis=0), osm_X

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

    def generate_data(self, ID_temp, channels, data):
        # load dataset statistics and patches 
        with rasterio.open(ID_temp, 'r') as ds: 
            if 'sen2' in data:
                if ds.shape!=(img_rows, img_cols):
                    image = ds.read((4,3,2), out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.cubic)
                else:
                    image = ds.read((4,3,2)) 
            elif data == 'lcz':
                image = ds.read(out_shape=(ds.count, img_rows, img_cols))
            elif "Pop" in data:
                image = ds.read(1)
            elif "sen1" in data:
                image = ds.read((1,2)) 
            else:
                image = ds.read(out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.average)

        if "sen2" in data:
            new_arr = ((image.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
        if "sen1" in data:
            new_arr = ((image.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
        elif data=="viirs":
            image = np.where(image < 0, 0, image)
            new_arr = ((image.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
        elif data=="lu":
            new_arr = ((image.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
        elif "Pop" in data:
            new_arr = image
        else: 
            new_arr = np.empty([channels, img_rows, img_cols])
            for k, layer in enumerate(image):
                if data == 'lcz':
                    arr = layer
                    arr = np.where((arr > 0) & (arr <= 10), arr * (-0.09) + 1.09, arr)
                    arr = np.where(arr == 0, 0.1, arr)
                    arr = np.where(arr > 10, 0, arr)
                    new_arr[k] = arr
                elif 'sen2' in data:
                    p2 = self.dataset_stats[data]['p2'][k]
                    mean = self.dataset_stats[data]['mean'][k]
                    std = self.dataset_stats[data]['std'][k]
                    arr = layer
                    arr = np.where(arr >= p2, p2, arr)
                    arr = arr - mean
                    arr = arr / std
                    new_arr[k] = arr
                elif data == 'viirs':
                    p2 = self.dataset_stats[data]['p2'][k]
                    mean = self.dataset_stats[data]['mean'][k]
                    std = self.dataset_stats[data]['std'][k]
                    arr = layer
                    arr = np.where(arr < 0, 0, arr)
                    arr = np.where(arr >= p2, p2, arr)
                    arr = arr - mean
                    arr = arr / std
                    new_arr[k] = arr
                else:
                    channel_min = self.dataset_stats[data]['min'][k]
                    channel_max = self.dataset_stats[data]['max'][k]
                    arr = layer
                    arr = arr - channel_min
                    arr = arr / (channel_max - channel_min)
                    new_arr[k] = arr
                # X = new_arr

        return new_arr


    def generate_osm_data(self, X, ID_temp, mm_scaler, channels):
        # Generate data
        # load csv
        df = pd.read_csv(ID_temp, header=None)[1]
        df = df[df.notna()]

        df_array = np.array(df)
        df_array[df_array == np.inf] = 0

        new_arr = np.empty([channels, osm_features])
        new_arr[0] = df_array

        # fit and transform the data
        new_arr = mm_scaler.transform(new_arr)
        scaled_arr = np.empty([channels, osm_features])
        scaled_arr[0] = new_arr
        return np.transpose(scaled_arr, (1, 0))
 
