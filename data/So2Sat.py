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
import random


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

        self.all_ids = list_IDs + f_names_unlab
        self.labeled_indices = [i for i,name in enumerate(self.all_ids) if name in list_IDs]
        self.unlabeled_indices = [i for i,name in enumerate(self.all_ids) if name in f_names_unlab]
 
        self.y_stats = load_json(os.path.join(config_path, 'dataset_stats', 'label_stats.json'))
        self.y_stats['max'] = float(self.y_stats['max'])
        self.y_stats['min'] = float(self.y_stats['min'])
        self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats_unified.json'))
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
            for idx in tqdm(range(len(self.all_ids))):
                # self.all_samples[self.list_IDs[idx]] = self[idx]
                self.all_samples[self.all_ids[idx]] = self[idx]
            self.move_to_memory = False 
            print("Done Loading to memory for Dataset: ", mode)

    def __getitem__(self, idx):
        if self.in_memory and not self.move_to_memory:
            sample = self.all_samples[self.all_ids[idx]]

            if self.transform:
                sample["input"] = self.transform(sample["input"])

            return sample

        # query the data ID
        ID_temp = self.all_ids[idx]

        # Generate data
        if self.satmode:
            X, Pop_X, PopNN_X, pop_avail, msb, msb_avail = self.data_generation(ID_temp)
        else:
            X, osm = self.data_generation(ID_temp)

        ID = ID_temp.split(os.sep)[-1].split('_sen2')[0]

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
        X = torch.from_numpy(X).type(torch.FloatTensor)
        Pop_X = torch.from_numpy(np.asarray(Pop_X)).type(torch.FloatTensor)
        PopNN_X = torch.from_numpy(np.asarray(PopNN_X)).type(torch.FloatTensor)
        pop_avail = torch.from_numpy(np.asarray(pop_avail)).type(torch.FloatTensor)
        msb = torch.from_numpy(np.asarray(msb)).type(torch.FloatTensor) 
        msb_avail = torch.from_numpy(np.asarray(msb_avail)).type(torch.FloatTensor)

        if self.move_to_memory:
            return {'input': X, 'y': y, 'y_norm': y_norm, 
                      'Pop_X': Pop_X, 'PopNN_X': PopNN_X, 'pop_avail': pop_avail,
                      'builtupmap': msb,  'msb_avail': msb_avail,
                    'identifier': ID, 'source': source}

        if self.transform:
            X = self.transform(X)

        if self.satmode:
            sample = {'input': X, 'y': y, 'y_norm': y_norm, 
                      'Pop_X': Pop_X, 'PopNN_X': PopNN_X, 'pop_avail': pop_avail,
                      'builtupmap': msb,  'msb_avail': msb_avail,
                      'identifier': ID, 'source': source}
        else:
            sample = {'input': X, 'y': y, 'y_norm': y_norm, 'osm': osm, 'identifier': ID}

        return sample


    def __len__(self):
        if self.mode=="train":
            return len(self.labels)
        else:
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
                if self.random_season:
                    ID_sen2 = ID_temp.replace('sen2spring', "sen2{}".format(random.choice(['spring', 'autumn', 'winter', 'summer'])))
                else:
                    ID_sen2 = ID_temp.replace('sen2spring', 'sen2spring') # for testing just use the spring images
                     
                ID_sen1 = ID_temp.replace('sen2spring', 'S1')
                ID_viirs = ID_temp.replace('sen2spring', 'viirs')
                ID_osm = ID_temp.replace('sen2spring', 'osm_features').replace('tif', 'csv')
                ID_lu = ID_temp.replace('sen2spring', 'lu')
                ID_Pop = ID_temp.replace('sen2spring', 'Pop')
                ID_PopNN = ID_temp.replace('sen2spring', 'PopNN')
                ID_msb = ID_temp.replace('sen2spring', 'msb')
                
                data = []
                if self.S2:
                    if self.NIR:
                        data.append(self.generate_data(ID_sen2, channels=4, data='sen2spring'))
                    else:
                        data.append(self.generate_data(ID_sen2, channels=3, data='sen2spring'))
                if self.S1:
                    data.append(self.generate_data(ID_sen1, channels=2 , data='sen1'))
                if self.VIIRS:
                    data.append(self.generate_data(ID_viirs, channels=1, data='viirs'))
                # osm_X = self.generate_osm_data(osm_X, ID_osm, mm_scaler, channels=1)
                # lu_X = self.generate_data(ID_lu, channels=4, data='lu') 

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

                use_msb = False
                if isfile(ID_msb) and use_msb:
                    msb = self.generate_data(ID_msb, channels=1, data='msb')
                    msb_avail = np.array([1])
                else:
                    msb = np.zeros((100,100))
                    msb_avail = np.array([0])
                
                return np.concatenate(data, axis=0), Pop_X, PopNN_X, pop_avail, msb, msb_avail

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
                if channels==3:
                    image = ds.read((4,3,2)) 
                    # image = ds.read((4,3,2), out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.cubic)
                elif channels==4:
                    image = ds.read((4,3,2,8)) 
            elif data == 'lcz':
                image = ds.read(out_shape=(ds.count, img_rows, img_cols))
            elif "Pop" in data:
                image = ds.read(1)
            elif "sen1" in data:
                image = ds.read((1,2)) 
            else:
                image = ds.read(out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.average)

        if "sen2" in data:
            if channels==3:
                image = np.where(image > self.dataset_stats["sen2spring"]['p2'][:,None,None], self.dataset_stats["sen2spring"]['p2'][:,None,None], image)
                new_arr = ((image.transpose((1,2,0)) - self.dataset_stats["sen2spring"]['mean'] ) / self.dataset_stats["sen2spring"]['std']).transpose((2,0,1))
            elif channels==4:
                image = np.where(image > self.dataset_stats["sen2springNIR"]['p2'][:,None,None], self.dataset_stats["sen2springNIR"]['p2'][:,None,None], image)
                new_arr = ((image.transpose((1,2,0)) - self.dataset_stats["sen2springNIR"]['mean'] ) / self.dataset_stats["sen2springNIR"]['std']).transpose((2,0,1))

        elif "sen1" in data:
            new_arr = ((image.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
        elif data=="viirs":
            image = np.where(image < 0, 0.0, image)
            new_arr = ((image.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
        elif data=="lu":
            new_arr = ((image.transpose((1,2,0)) - self.dataset_stats[data]['mean'] ) / self.dataset_stats[data]['std']).transpose((2,0,1))
        elif "Pop" in data:
            new_arr = image
        else: 
            raise ValueError("Invalid data type")

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
 
