import os
import glob
import csv
import random

import numpy as np
import torch
from tqdm import tqdm
from random import randrange
import time
from pylab import figure, imshow, matshow, grid, savefig, colorbar
import pandas as pd
import matplotlib.pyplot as plt
from model.pomelo import JacobsUNet, PomeloUNet, ResBlocks, UResBlocks, ResBlocksDeep, ResBlocksSqueeze
from model.ownmodels import BoostUNet
import json


def to_cuda(sample):
    sampleout = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            sampleout[key] = val.cuda()
        elif isinstance(val, list):
            new_val = []
            for e in val:
                if isinstance(e, torch.Tensor):
                    new_val.append(e.cuda())
                else:
                    new_val.append(val)
            sampleout[key] = new_val
        else:
            sampleout[key] = val
    return sampleout

def to_cuda_inplace(sample):
    # sampleout = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            sample[key] = val.cuda()
        elif isinstance(val, list):
            new_val = []
            for e in val:
                if isinstance(e, torch.Tensor):
                    new_val.append(e.cuda())
                else:
                    new_val.append(val)
            sample[key] = new_val
        elif isinstance(val, dict):
            sample[key] = to_cuda_inplace(val)
        else:
            sample[key] = val
    return sample

def detach_tensors_in_dict(input_dict):
    return {key: value.detach() if torch.is_tensor(value) else value for key, value in input_dict.items()}

def seed_all(seed):
    # Fix all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def new_log(folder_path, args=None):
    os.makedirs(folder_path, exist_ok=True)
    n_exp = len(os.listdir(folder_path))
    randn  = round((time.time()*1000000) % 1000)
    experiment_folder = os.path.join(folder_path, f'experiment_{n_exp}_{randn}')
    os.mkdir(experiment_folder)

    if args is not None:
        args_dict = args.__dict__
        write_params(args_dict, os.path.join(experiment_folder, 'args' + '.csv'))

    return experiment_folder, n_exp, randn


def write_params(params, path):
    with open(path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['key', 'value'])
        for data in params.items():
            writer.writerow([el for el in data])


def get_fnames_labs_reg(path, force_recompute=False):
    """
    :param path: path to patch folder (sen2spring), now S2Aspring
    :return: gives the paths of all the tifs and its corresponding class labels
    """

    fnames_file = os.path.join(path, 'file_list.txt')
    labs_file = os.path.join(path, 'label_list.txt')

    if os.path.isfile(fnames_file) and os.path.isfile(labs_file) and (not force_recompute):
        # read filenames from file, Define an empty list
        f_names_all = []
        labs_all = []

        # Open the file and read the content in a list
        with open(fnames_file, 'r') as filehandle:
            for line in filehandle:
                curr_place = line[:-1] # Remove linebreak which is the last character of the string
                f_names_all.append(os.path.join(path, curr_place))

        with open(labs_file, 'r') as filehandle:
            for line in filehandle:
                curr_place = line[:-1] # Remove linebreak which is the last character of the string
                labs_all.append(float(curr_place))

    else:
        city_folders = glob.glob(os.path.join(path, "*"))
        f_names_save_all = np.array([])
        f_names_all = np.array([])
        labs_all = np.array([])
        for each_city in tqdm(city_folders):
            if each_city.endswith(".txt"):
                continue
            # data_path = os.path.join(each_city, "sen2spring")
            data_path = os.path.join(each_city, "S2Aspring")
            csv_path = os.path.join(each_city, each_city.split(os.sep)[-1:][0] + '.csv')
            city_df = pd.read_csv(csv_path)
            ids = city_df['GRD_ID']
            pop = city_df['POP']
            classes = city_df['Class']
            classes_str = [str(x) for x in classes]
            classes_paths = [data_path + '/Class_' + x + '/' for x in classes_str]
            for index in range(0, len(classes_paths)):
                # f_names = [classes_paths[index] + str(ids[index]) + '_sen2spring.tif']
                f_names = [classes_paths[index] + str(ids[index]) + '_S2Aspring.tif']
                f_names_all = np.append(f_names_all, f_names, axis=0)
                # f_names_save = [(classes_paths[index] + str(ids[index]) + '_sen2spring.tif').split(path+"/")[1]]
                f_names_save = [(classes_paths[index] + str(ids[index]) + '_S2Aspring.tif').split(path+"/")[1]]
                f_names_save_all = np.append(f_names_save_all, f_names_save, axis=0)
                labs = [pop[index]]
                labs_all = np.append(labs_all, labs, axis=0)

        # Write the found lists to the disk to later load it more quickly
        with open(fnames_file, 'w') as filehandle1:
            with open(labs_file, 'w') as filehandle2:
                for fname, la in zip(f_names_save_all, labs_all):
                    filehandle1.write(f'{fname}\n')
                    filehandle2.write(f'{la}\n')

        f_names_all = f_names_all.tolist()
        labs_all = labs_all.tolist()

    return f_names_all, labs_all

def get_fnames_unlab_reg(parent_dir, force_recompute=False):
    """
    :param parent_dir: path to patch folder (sen2spring), now S2Aspring
    :return: gives the paths of all the tifs and its corresponding class labels
    """
 
    # data_path = os.path.join(parent_dir, "sen2spring")
    data_path = os.path.join(parent_dir, "S2Aspring")
    fnames_file = os.path.join(parent_dir, 'file_list.txt')
    # labs_file = os.path.join(parent_dir, 'label_list.txt')

    if os.path.isfile(fnames_file) and (not force_recompute):
        # read filenames from file, Define an empty list
        f_names_all = []

        # Open the file and read the content in a list
        with open(fnames_file, 'r') as filehandle:
            for line in filehandle:
                curr_place = line[:-1] # Remove linebreak which is the last character of the string
                f_names_all.append(os.path.join(parent_dir, curr_place))

    else:
        # iterate though the data path and list names
        f_names_all = []
        files_save_all = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".tif"):
                    f_name = os.path.join(root, file)
                    f_names_all.append(f_name)
                    files_save_all.append(f_name.split(parent_dir+"/")[1])

        # Write the found lists to the disk to later load it more quickly
        with open(fnames_file, 'w') as filehandle1:
            for fname in files_save_all:
                filehandle1.write(f'{fname}\n')

    return f_names_all

# 
model_dict = {
    "JacobsUNet": JacobsUNet,
    "PomeloUNet": PomeloUNet,
    "ResBlocks": ResBlocks,
    "ResBlocksSqueeze": ResBlocksSqueeze,
    "UResBlocks": UResBlocks,
    "ResBlocksDeep": ResBlocksDeep,
    "BoostUNet": BoostUNet,
}


def get_model_kwargs(args, model_name):
    """
    :param args: arguments
    :param model_name: name of the model
    :return: kwargs for the model
    """

    # kwargs for the model
    kwargs = {
        'input_channels': args.Sentinel1 * 2 + args.NIR * 1 + args.Sentinel2 * 3 + args.VIIRS * 1,
        'feature_dim': args.feature_dim,
        'feature_extractor': args.feature_extractor
    }

    # additional kwargs for the Jacob's model
    if model_name == 'JacobsUNet':
        kwargs['classifier'] = args.classifier if args.adversarial else None
        kwargs['head'] = args.head
        kwargs['down'] = args.down
    if model_name == 'BoostUNet':
        assert args.Sentinel1
        assert args.Sentinel2
        kwargs['classifier'] = args.classifier if args.adversarial else None
        # kwargs['head'] = args.head
        kwargs['down'] = args.down
        kwargs['down2'] = args.down2
        kwargs['occupancymodel'] = args.occupancymodel 
        kwargs['useallfeatures'] = args.useallfeatures
    return kwargs


def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def apply_normalize(indata, dataset_stats):

    # S2
    if "S2" in indata:
        if indata["S2"].shape[0] == 4:
            # indata["S2"] = torch.where(indata["S2"] > self.dataset_stats["sen2springNIR"]['p2'][:,None,None], self.dataset_stats["sen2springNIR"]['p2'][:,None,None], indata["S2"])
            indata["S2"] = ((indata["S2"].permute((0,2,3,1)) - dataset_stats["sen2springNIR"]['mean'] ) / dataset_stats["sen2springNIR"]['std']).permute((0,3,1,2))
        else: 
            # indata["S2"] = torch.where(indata["S2"] > self.dataset_stats["sen2spring"]['p2'][:,None,None], self.dataset_stats["sen2spring"]['p2'][:,None,None], indata["S2"])
            # indata["S2"] = ((indata["S2"].permute((1,2,0)) - dataset_stats["sen2spring"]['mean'] ) / dataset_stats["sen2spring"]['std']).permute((2,0,1))
            indata["S2"] = ((indata["S2"].permute((0,2,3,1)) - dataset_stats["sen2spring"]['mean'].cuda() ) / dataset_stats["sen2spring"]['std'].cuda()).permute((0,3,1,2))

    # S1
    if "S1" in indata:
        # indata["S1"] = torch.where(indata["S1"] > self.dataset_stats["sen1"]['p2'][:,None,None], self.dataset_stats["sen1"]['p2'][:,None,None], indata["S1"])
        indata["S1"] = ((indata["S1"].permute((0,2,3,1)) - dataset_stats["sen1"]['mean'].cuda() ) / dataset_stats["sen1"]['std'].cuda()).permute((0,3,1,2))

    # VIIRS
    if "VIIRS" in indata:
        # indata["VIIRS"] = torch.where(indata["VIIRS"] > self.dataset_stats["viirs"]['p2'][:,None,None], self.dataset_stats["viirs"]['p2'][:,None,None], indata["VIIRS"])
        indata["VIIRS"] = ((indata["VIIRS"].permute((0,3,1,2)) - dataset_stats["viirs"]['mean'].cuda() ) / dataset_stats["viirs"]['std'].cuda()).permute((0,3,1,2))

    return indata


def apply_transformations_and_normalize(sample, transform, dataset_stats):
    """
    :param image: image to be transformed
    :param mask: mask to be transformed
    :param transform: transform to be applied to the image
    :param transform_mask: transform to be applied to the mask
    :return: transformed image and mask
    """
    # transforms

    # Modality-wise transformations
    if transform is not None:
        if "S2" in transform and "S2" in sample:
            sample["S2"] = transform["S2"](sample["S2"])
        if "S1" in transform and "S1" in sample:
            sample["S1"] = transform["S1"](sample["S1"])
        if "VIIRS" in transform and "VIIRS" in sample:
            sample["VIIRS"] = transform["VIIRS"](sample["VIIRS"])

    # Normalizations
    sample = apply_normalize(sample, dataset_stats)
    # sample = normalize_indata(sample, normalization)
    
    # merge inputs
    sample["input"] = torch.concatenate([sample[key] for key in ["S2", "S1", "VIIRS"] if key in sample], dim=1)

    # General transformations
    if transform is not None:
        # apply the transformation to the image

        if "general" in transform.keys():
            # if masked
            if "mask" in sample.keys(): 
                sample["input"], sample["mask"] = transform["general"]((sample["input"], sample["mask"])) 
            else:
                sample["input"] = transform["general"](sample["input"])
            
    

    return sample