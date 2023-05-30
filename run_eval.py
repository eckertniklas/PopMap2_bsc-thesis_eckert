import os
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
from torch import is_tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ChainDataset, ConcatDataset
from torchvision.transforms import Normalize
from torchvision import transforms
from utils.transform import OwnCompose
from utils.transform import RandomRotationTransform, RandomHorizontalFlip, RandomVerticalFlip, RandomHorizontalVerticalFlip, RandomBrightness, RandomGamma, HazeAdditionModule, AddGaussianNoise
from tqdm import tqdm

import itertools
import random
from sklearn import model_selection
import wandb

import pickle
import gc

# from arguments import eval_parser
from arguments.eval import parser as eval_parser
from data.So2Sat import PopulationDataset_Reg
from data.PopulationDataset_target import Population_Dataset_target, Population_Dataset_collate_fn
from utils.losses import get_loss, r2
from utils.metrics import get_test_metrics
from utils.utils import new_log, to_cuda, to_cuda_inplace, detach_tensors_in_dict, seed_all, get_model_kwargs, model_dict

from utils.plot import plot_2dmatrix, plot_and_save, scatter_plot3
from utils.utils import get_fnames_labs_reg, get_fnames_unlab_reg
from utils.datasampler import LabeledUnlabeledSampler
from utils.constants import img_rows, img_cols, all_patches_mixed_train_part1, all_patches_mixed_test_part1, pop_map_root, inference_patch_size, overlap
from utils.constants import inference_patch_size as ips

import nvidia_smi
nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # set up dataloaders
        self.dataloaders = self.get_dataloaders(args)
        
        # set up model
        seed_all(args.seed)

        # define architecture
        if args.model in model_dict:
            model_kwargs = get_model_kwargs(args, args.model)
            self.model = model_dict[args.model](**model_kwargs).cuda()
        else:
            raise ValueError(f"Unknown model: {args.model}")

        # number of params
        args.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Model", args.model, "; #Params:", args.pytorch_total_params)

        # set up experiment folder
        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, "So2Sat"), args)
        self.args.experiment_folder = self.experiment_folder

        # wandb config
        wandb.init(project=args.wandb_project, dir=self.experiment_folder)
        wandb.config.update(self.args) 

        # in case of checkpoint resume
        if args.resume is not None:
            self.resume(path=args.resume)

    def test_target(self, save=False):
        # Test on target domain
        self.model.eval()
        self.test_stats = defaultdict(float)

        with torch.no_grad(): 
            for testdataloader in self.dataloaders["test_target"]:

                # inputialize the output map
                h, w = testdataloader.dataset.shape()
                output_map = torch.zeros((h, w))
                output_map_var = torch.zeros((h, w))
                output_map_count = torch.zeros((h, w))

                for sample in tqdm(testdataloader, leave=False):
                    sample = to_cuda_inplace(sample)

                    # get the valid coordinates
                    xmin, xmax, ymin, ymax = [val.item() for val in sample["valid_coords"]]
                    xl,yl = [val.item() for val in sample["img_coords"]]
                    mask = sample["mask"][0].bool()

                    # get the output with a forward pass
                    output = self.model(sample, padding=False)
                    if mask.sum()<(921600-1):
                        print("sus...")
                    output_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["popdensemap"][0][mask].cpu()
                    output_map_var[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["popvarmap"][0][mask].cpu()
                    output_map_count[xl:xl+ips, yl:yl+ips][mask.cpu()] += 1

                # average over the number of times each pixel was visited
                output_map[output_map_count>0] = output_map[output_map_count>0] / output_map_count[output_map_count>0]
                output_map_var[output_map_count>0] = output_map_var[output_map_count>0] / output_map_count[output_map_count>0]

                if save:
                    # save the output map
                    testdataloader.dataset.save(output_map, self.experiment_folder)
                    testdataloader.dataset.save(output_map_var, self.experiment_folder, tag="VAR")
                
                # convert populationmap to census
                census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map, gpu_mode=True)
                self.target_test_stats = get_test_metrics(census_pred, census_gt.float().cuda(), tag="census")
                built_up = census_gt>10
                self.target_test_stats = {**self.target_test_stats,
                                          **get_test_metrics(census_pred[built_up], census_gt[built_up].float().cuda(), tag="census_pos")}
                
                scatterplot = scatter_plot3(census_pred.tolist(), census_gt.tolist())
                if scatterplot is not None:
                    self.target_test_stats["scatter_PRI"] = wandb.Image(scatterplot)
                    scatterplot.save("last_scatter.png")

                wandb.log({**{k + '/targettest': v for k, v in self.target_test_stats.items()}})
                print(self.target_test_stats)
        

    @staticmethod
    def get_dataloaders(args): 
        """
        Get dataloaders for the source and target domains
        Inputs:
            args: command line arguments
            force_recompute: if True, recompute the dataloader's and look out for new files even if the file list already exist
        Outputs:
            dataloaders: dictionary of dataloaders
                """

        input_defs = {'S1': args.Sentinel1, 'S2': args.Sentinel2, 'VIIRS': args.VIIRS, 'NIR': args.NIR}

        # create the raw source dataset
        datasets = {
            "test_target": [ Population_Dataset_target(reg, patchsize=ips, overlap=overlap, fourseasons=True,**input_defs) for reg in args.target_regions ]
        }
        
        # create the dataloaders
        dataloaders =  {
            "test_target":  [DataLoader(datasets["test_target"], batch_size=1, num_workers=1, shuffle=False, drop_last=False) for datasets["test_target"] in datasets["test_target"] ]
        }
        
        return dataloaders


    def resume(self, path):
        """
        Input:
            path: path to the checkpoint
        """
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        # load checkpoint
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])
        # self.info["epoch"] = checkpoint['epoch']
        # self.info["iter"] = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = eval_parser.parse_args()
    print(eval_parser.format_values())

    trainer = Trainer(args)

    since = time.time() 
    trainer.test_target(save=True)
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))