import json
import os

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
    def __init__(self, list_IDs, labels, dim=(img_rows, img_cols), transform=None, test=False, mode=None, satmode=False, in_memory=False): 

        self.dim = dim
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = num_classes
        self.transform = transform
        self.test = test
        self.mode = mode 
        self.in_memory = in_memory
        self.move_to_memory = False
                                                    
        self.satmode = satmode
        self.y_stats = load_json(os.path.join(config_path, 'dataset_stats', 'label_stats.json'))
        self.y_stats['max'] = float(self.y_stats['max'])
        self.y_stats['min'] = float(self.y_stats['min'])
        self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'mod_dataset_stats.json'))
        for mkey in self.dataset_stats.keys():
            if isinstance(self.dataset_stats[mkey], dict):
                for key,val in self.dataset_stats[mkey].items():
                    self.dataset_stats[mkey][key] = np.array(val)
            else:
                self.dataset_stats[mkey] = np.array(val)

        self.all_samples = {}
        if in_memory:
            print("Loading to memory for Dataset: ", mode)
            self.move_to_memory = True 
            # quickdataloader = DataLoader(self, batch_size=1, num_workers=4, shuffle=True, drop_last=False)
            # for idx,sample in enumerate(tqdm(quickdataloader)):
            #     self.all_samples[self.list_IDs[idx]] = sample
            # del quickdataloader

            
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
            X, osm, msb = self.data_generation(ID_temp)
        else:
            X, osm = self.data_generation(ID_temp)

        ID = ID_temp.split(os.sep)[-1].split('_sen2')[0]
        y = self.labels[idx]
        y_norm = self.normalize_reg_labels(y) 

        # To torch
        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(np.asarray(y)).type(torch.FloatTensor)
        y_norm = torch.from_numpy(np.asarray(y_norm)).type(torch.FloatTensor)
        osm = torch.from_numpy(osm).type(torch.FloatTensor)
        msb = torch.from_numpy(msb).type(torch.FloatTensor)

        if self.move_to_memory:
            return {'input': X, 'y': y, 'y_norm': y_norm, 'builtupmap': msb, 'osm': osm, 'identifier': ID}

        if self.transform:
            X = self.transform(X)

        if self.satmode:
            sample = {'input': X, 'y': y, 'y_norm': y_norm, 'builtupmap': msb, 'osm': osm, 'identifier': ID}
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
                ID_viirs = ID_temp.replace('sen2spring', 'viirs')
                ID_osm = ID_temp.replace('sen2spring', 'osm_features').replace('tif', 'csv')
                ID_lu = ID_temp.replace('sen2spring', 'lu')
                # dem_X = generate_data(dem_X, ID_dem, channels=1, data='dem')
                # TODO: check if MS buildings exist, and load them
                # TODO: load S1 imagery

                sen2spr_X = self.generate_data(ID_spring, channels=3, data='sen2spring')
                viirs_X = self.generate_data(ID_viirs, channels=1, data='viirs')  
                # osm_X = self.generate_osm_data(osm_X, ID_osm, mm_scaler, channels=1)
                lu_X = self.generate_data(ID_lu, channels=4, data='lu') 

                return np.concatenate((sen2spr_X, viirs_X), axis=0), np.array([0]) , np.argmax(lu_X,0)>1.5
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
                # image = image[::-1, :, :]

                # image = ds.read(out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.cubic)
                # print('Band1 has shape', band1.shape)
                # height = image.shape[1]
                # width = image.shape[2]
                # cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                # xs, ys = rasterio.transform.xy(ds.transform, rows, cols)

                # if image.shape[0] == 13:
                    ## for sentinel-2 images, reading only the RGB bands
                    # image = image[1:4]
                    # image = image[::-1, :, :]
            elif data == 'lcz':
                image = ds.read(out_shape=(ds.count, img_rows, img_cols))
            else:
                image = ds.read(out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.average)

        if "sen2" in data:
            new_arr = ((image.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
        elif data=="viirs":
            image = np.where(image < 0, 0, image)
            new_arr = ((image.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
        elif data=="lu":
            new_arr = ((image.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
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
 
