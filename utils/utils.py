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


def plot_2dmatrix(matrix, fig=1, vmin=None, vmax=None):
    if torch.is_tensor(matrix):
        if matrix.is_cuda:
            matrix = matrix.cpu()
        if matrix.requires_grad:
            matrix = matrix.detach()
        matrix = matrix.numpy()
    if matrix.shape[0]==1:
        matrix = matrix[0]
    if matrix.shape[0]==3:
        matrix = matrix.transpose((1,2,0))

    figure(fig)
    matshow(matrix, interpolation='nearest', vmin=vmin, vmax=vmax)
    grid(True)
    colorbar()
    savefig('plot_outputs/last_plot.png')


def plot_and_save(img, mask=None, vmax=None, vmin=None, idx=None,
    model_name='model_name', title=None, name='latest_figure', colorbar=True, cmap="viridis"):

    if mask is not None:
        img = np.ma.masked_where(mask, img)

    # convert img tensor to uint8
    
    plt.figure(figsize=(12, 8), dpi=260)
    plt.imshow((img * 255).to(torch.uint8), vmax=vmax, vmin=vmin, cmap=cmap)
    if colorbar:
        plt.colorbar()
    title = title if title is not None else model_name
    plt.title(title)

    if idx is not None:
        if not os.path.exists(f'vis/{model_name}/'):
            os.makedirs(f'vis/{model_name}/')
        plt.savefig(f'vis/{model_name}/{str(idx).zfill(4)}_{name}.png')
        plt.close()
    else:
        plt.savefig(f'{name}.png')


def get_fnames_labs_reg(path, force_recompute=False):
    """
    :param path: path to patch folder (sen2spring)
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
                    f_names_all.append(curr_place)

        with open(labs_file, 'r') as filehandle:
                for line in filehandle:
                    curr_place = line[:-1] # Remove linebreak which is the last character of the string
                    labs_all.append(float(curr_place))
    
    else:
        city_folders = glob.glob(os.path.join(path, "*"))
        f_names_all = np.array([])
        labs_all = np.array([])
        for each_city in tqdm(city_folders):
            if each_city.endswith(".txt"):
                continue
            data_path = os.path.join(each_city, "sen2spring")
            csv_path = os.path.join(each_city, each_city.split(os.sep)[-1:][0] + '.csv')
            city_df = pd.read_csv(csv_path)
            ids = city_df['GRD_ID']
            pop = city_df['POP']
            classes = city_df['Class']
            classes_str = [str(x) for x in classes]
            classes_paths = [data_path + '/Class_' + x + '/' for x in classes_str]
            for index in range(0, len(classes_paths)):
                f_names = [classes_paths[index] + str(ids[index]) + '_sen2spring.tif']
                f_names_all = np.append(f_names_all, f_names, axis=0)
                labs = [pop[index]]
                labs_all = np.append(labs_all, labs, axis=0)

        # Write the found lists to the disk to later load it more quickly
        with open(fnames_file, 'w') as filehandle1:
            with open(labs_file, 'w') as filehandle2:
                for fname, la in zip(f_names_all,labs_all):
                    filehandle1.write(f'{fname}\n')
                    filehandle2.write(f'{la}\n')

    return f_names_all, labs_all

def get_fnames_unlab_reg(parent_dir, force_recompute=False):
 
    data_path = os.path.join(parent_dir, "sen2spring")
    fnames_file = os.path.join(parent_dir, 'file_list.txt')
    # labs_file = os.path.join(parent_dir, 'label_list.txt')

    if os.path.isfile(fnames_file) and (not force_recompute):
        # read filenames from file, Define an empty list
        f_names_all = []

        # Open the file and read the content in a list
        with open(fnames_file, 'r') as filehandle:
            for line in filehandle:
                curr_place = line[:-1] # Remove linebreak which is the last character of the string
                f_names_all.append(curr_place)

    else:
        # iterate though the data path and list names
        f_names_all = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".tif"):
                    f_names_all.append(os.path.join(root, file))

            # Write the found lists to the disk to later load it more quickly
            with open(fnames_file, 'w') as filehandle1:
                for fname in f_names_all:
                    filehandle1.write(f'{fname}\n')

    return f_names_all
