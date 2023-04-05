

import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import random

from typing import Dict, Tuple

from utils.constants import pop_map_root, pop_map_covariates, config_path

from utils.utils import plot_2dmatrix

def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


class Population_Dataset_target(Dataset):
    """
    Population dataset for target domain
    Use this dataset to evaluate the model on the target domain and compare it the census data
    """
    def __init__(self, region, S1=False, S2=True, VIIRS=True, NIR=False, patchsize=1024, overlap=32, fourseasons=False, mode="test") -> None:
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

        # get the path to the data
        region_root = os.path.join(pop_map_root, region)

        # load the boundary and census data
        self.boundary_file = os.path.join(region_root, "boundaries4.tif")
        self.census_file = os.path.join(region_root, "census4.csv")
        self.coarse_boundary_file = os.path.join(region_root, "boundaries_COUNTYFP20.tif")
        self.coarse_census_file = os.path.join(region_root, "census_COUNTYFP20.csv")
        
        if self.mode == "weaksup":
            # weaksup data specific preparation
            # read the census file
            self.coarse_census = pd.read_csv(self.coarse_census_file)
            max_pix = 2e6
            print("Kicking out ", (self.coarse_census["count"]>=max_pix).sum(), "samples with more than ", int(max_pix), " pixels")
            self.coarse_census = self.coarse_census[self.coarse_census["count"]<max_pix].reset_index()
            # redefine indexing

            # get the shape of the coarse regions
            with rasterio.open(self.coarse_boundary_file, "r") as src:
                self.cr_regions = src.read(1)

        elif self.mode=="test":
            # testdata specific preparation
            # get the shape of the images
            with rasterio.open(self.boundary_file, "r") as src:
                self.img_shape = src.shape
                self._meta = src.meta.copy()
            self._meta.update(count=1, dtype='float32')

            # get a list of indices of the possible patches
            self.patch_indices = self.get_patch_indices(patchsize, overlap)

        # get the path to the data files
        covar_root = os.path.join(pop_map_covariates, region)
        self.S1_file = os.path.join(covar_root,  os.path.join("S1", region +"_S1.tif"))
        self.sen2spring_file = os.path.join(covar_root,  os.path.join("sen2spring", region +"_sen2spring.tif"))
        self.sen2summer_file = os.path.join(covar_root,  os.path.join("sen2summer", region +"_sen2summer.tif"))
        self.sen2autumn_file = os.path.join(covar_root,  os.path.join("sen2autumn", region +"_sen2autumn.tif"))
        self.sen2winter_file = os.path.join(covar_root,  os.path.join("sen2winter", region +"_sen2winter.tif"))
        self.S2_file = {0: self.sen2spring_file, 1: self.sen2summer_file, 2: self.sen2autumn_file, 3: self.sen2winter_file}
        self.season_dict = {0: "spring", 1: "summer", 2: "autumn", 3: "winter"}
        self.inv_season_dict = {v: k for k, v in self.season_dict.items()}
        self.VIIRS_file = os.path.join(covar_root,  os.path.join("viirs", region +"_viirs.tif"))

        # normalize the dataset (do not use, this does not make sense for variable regions sizes like here)
        self.y_stats = load_json(os.path.join(config_path, 'dataset_stats', 'label_stats.json'))
        self.y_stats['max'] = float(self.y_stats['max'])
        self.y_stats['min'] = float(self.y_stats['min'])

        # load the dataset stats
        self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats_unified.json'))
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
        data, _ = self.data_generation(xmin, ymin, self.inv_season_dict[season], patchsize=(xmax-xmin, ymax-ymin), overlap=0)

        # return dictionary
        return {'input': torch.from_numpy(data).type(torch.FloatTensor),
                'y': torch.from_numpy(np.asarray(census_sample["POP20"])).type(torch.FloatTensor),
                'admin_mask': torch.from_numpy(self.cr_regions[xmin:xmax, ymin:ymax]).type(torch.FloatTensor),
                'img_coords': (xmin,ymin), 'valid_coords':  (xmin, xmax, ymin, ymax),
                'season': self.inv_season_dict[season],# 'season_str': [season],
                'source': True, "census_idx": census_sample["idx"],
                }


    def __gettestitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        # get the indices of the patch
        x,y,season = self.patch_indices[index]
        data, mask= self.data_generation(x,y,season.item())
        data = torch.from_numpy(data).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        # get valid coordinates of the patch
        xmin = x+self.overlap
        xmax = x+self.patchsize-self.overlap
        ymin = y+self.overlap
        ymax = y+self.patchsize-self.overlap
        
        # return dictionary
        return {'input': data, 'img_coords': (x,y), 'valid_coords':  (xmin, xmax, ymin, ymax),
                'season': season.item(), 'mask': mask, 'season_str': self.season_dict[season.item()]}
    

    def data_generation(self,x,y, season, patchsize=None, overlap=None) -> Tuple[np.ndarray, np.ndarray]:
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

        data = []
        mask = np.zeros((patchsize_x, patchsize_y), dtype=bool)
        mask[overlap:patchsize_x-overlap, overlap:patchsize_y-overlap] = True
        fake = False

        # get the input data
        if self.S2:
            S2_file = self.S2_file[season]
            if self.NIR:
                if fake:
                    raw_data = np.random.randint(0, 10000, size=(4,patchsize_x,patchsize_y))
                else:
                    with rasterio.open(S2_file, "r") as src:
                        raw_data = src.read((4,3,2,8), window=((x,x+patchsize_x),(y,y+patchsize_y))) 
                this_mask = raw_data.sum(axis=0) != 0
                mask = mask & this_mask
                raw_data = np.where(raw_data > self.dataset_stats["sen2springNIR"]['p2'][:,None,None], self.dataset_stats["sen2springNIR"]['p2'][:,None,None], raw_data)
                new_arr = ((raw_data.transpose((1,2,0)) - self.dataset_stats["sen2springNIR"]['mean'] ) / self.dataset_stats["sen2springNIR"]['std']).transpose((2,0,1))
                data.append(new_arr)
            else:
                if fake:
                    raw_data = np.random.randint(0, 10000, size=(3,patchsize_x,patchsize_y))
                    with rasterio.open(S2_file, "r") as src:
                        raw_data = src.read((4,3,2), window=((x,x+patchsize_x),(y,y+patchsize_y))) 
                this_mask = raw_data.sum(axis=0) != 0
                mask = mask & this_mask
                raw_data = np.where(raw_data > self.dataset_stats["sen2spring"]['p2'][:,None,None], self.dataset_stats["sen2spring"]['p2'][:,None,None], raw_data)
                new_arr = ((raw_data.transpose((1,2,0)) - self.dataset_stats["sen2spring"]['mean'] ) / self.dataset_stats["sen2spring"]['std']).transpose((2,0,1))
                data.append(new_arr)
        if self.S1:
            if fake:
                raw_data = np.random.randint(0, 10000, size=(2,patchsize_x,patchsize_y))
            else:
                with rasterio.open(self.S1_file, "r") as src:
                    raw_data = src.read(window=((x,x+patchsize_x),(y,y+patchsize_y))) 
            # raw_data = np.where(raw_data > self.dataset_stats["sen1"]['p2'][:,None,None], self.dataset_stats["sen1"]['p2'][:,None,None], raw_data)
            new_arr = ((raw_data.transpose((1,2,0)) - self.dataset_stats["sen1"]['mean'] ) / self.dataset_stats["sen1"]['std']).transpose((2,0,1))
            data.append(new_arr)
        if self.VIIRS:
            if fake:
                raw_data = np.random.randint(0, 10000, size=(1,patchsize_x,patchsize_y))
            else:
                with rasterio.open(self.VIIRS_file, "r") as src:
                    raw_data = src.read(window=((x,x+patchsize_x),(y,y+patchsize_y))) 
            raw_data = np.where(raw_data < 0, 0, raw_data)
            new_arr = ((raw_data.transpose((1,2,0)) - self.dataset_stats["viirs"]['mean'] ) / self.dataset_stats["viirs"]['std']).transpose((2,0,1))
            data.append(new_arr)

        return np.concatenate(data, axis=0), mask


    def convert_popmap_to_census(self, pred, gpu_mode=False):
        """
        Converts the predicted population to the census data
        inputs:
            :param pred: predicted population
            :param gpu_mode: if aggregation is done on gpu (can use a lot of memory, but is a lot faster)
        outputs:
            :return: the predicted population for each census region
        """

        # raise NotImplementedError
        with rasterio.open(self.boundary_file, "r") as src:
            boundary = src.read(1)

        # read the census file
        census = pd.read_csv(self.census_file)

        if gpu_mode:
            pred = pred.cuda()
            boundary = torch.from_numpy(boundary).cuda()

            # iterate over census regions and get totals
            census_pred = torch.zeros(len(census), dtype=torch.float32).cuda()
            # for i in tqdm(range(len(census))):
            for i,bbox in (zip(census["idx"], census["bbox"])):
                xmin, xmax, ymin, ymax = tuple(map(int, census.loc[i]["bbox"].strip('()').split(',')))
                # xmin, xmax, ymin, ymax = tuple(map(int, census.loc[3]["bbox"].strip('()').split(',')))
                census_pred[i] = pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==i].sum()

        else:
            boundary = torch.from_numpy(boundary)

            # iterate over census regions and get totals
            census_pred = torch.zeros(len(census), dtype=torch.float32)
            for i in tqdm(range(len(census))):
                xmin, xmax, ymin, ymax = tuple(map(int, census.loc[3]["bbox"].strip('()').split(',')))
                census_pred[i] = pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==i].sum()
        
        del boundary, pred
        torch.cuda.empty_cache()

        return census_pred, torch.tensor(census["POP20"])
    
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
    
    # taken from https://stackoverflow.com/questions/64022697/max-pooling-with-complex-masks-in-pytorch
    def mask_max_pool(self, embeddings, mask):
        '''
        Inputs:
        ------------------
        embeddings: [B, D, E], 
        mask: [B, R, D], 0s and 1s, 1 indicates membership

        Outputs:
        ------------------
        max pooled embeddings: [B, R, E], the max pooled embeddings according to the membership in mask
        max pooled index： [B, R, E], the max pooled index
        '''
        raise NotImplementedError
        B, D, E = embeddings.shape
        _, R, _ = mask.shape
        # extend embedding with placeholder
        embeddings_ = torch.cat([-1e6*torch.ones_like(embeddings[:, :1, :]), embeddings], dim=1)
        # transform mask to index
        index = torch.arange(1, D+1).view(1, 1, -1).repeat(B, R, 1) * mask# [B, R, D]
        # batch indices
        batch_indices = torch.arange(B).view(B, 1, 1).repeat(1, R, D)
        # retrieve embeddings by index
        indexed = embeddings_[batch_indices.flatten(), index.flatten(), :].view(B, R, D, E)# [B, R, D, E]
        # return
        return indexed.max(dim=-2)

    def save(self, preds, output_folder) -> None:
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
        output_file = os.path.join(output_folder, self.region + "_predictions.tif")
        with rasterio.open(output_file, "w", **self._meta) as dest:
            dest.write(preds,1)
        pass


if __name__=="__main__":

    #test the dataset
    from torch.utils.data import DataLoader, ChainDataset, ConcatDataset

    input_defs = {'S1': True, 'S2': True, 'VIIRS': False, 'NIR': True}

    dataset = Population_Dataset_target("pri2017", mode="weaksup", patchsize=None, overlap=None, fourseasons=True, **input_defs) 
    
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True, drop_last=True)


    # with tqdm(dataloader, leave=False) as inner_tnr:
    #     for i, sample in enumerate(inner_tnr):

    #         print(i, sample['input'].shape)
    
    for e in tqdm(range(10), leave=True):
        # unsupervised mode
        dataloader_iterator = iter(dataloader)
        
        for i in tqdm(range(5000)):

            sample = dataset[i%len(dataset)]
            print(i,sample['input'].shape)
            # try:
            #     sample_weak = next(dataloader_iterator)
            # except StopIteration:
            #     dataloader_iterator = iter(dataloader)
            #     sample_weak = next(dataloader_iterator)


