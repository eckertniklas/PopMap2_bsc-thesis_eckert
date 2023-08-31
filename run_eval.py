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
from utils.utils import new_log, to_cuda, to_cuda_inplace, detach_tensors_in_dict, seed_all
from model.get_model import get_model_kwargs, model_dict
from utils.utils import load_json, apply_transformations_and_normalize, apply_normalize
from utils.constants import config_path

from utils.plot import plot_2dmatrix, plot_and_save, scatter_plot3
# from utils.utils import get_fnames_labs_reg, get_fnames_unlab_reg
# from utils.datasampler import LabeledUnlabeledSampler
from utils.constants import img_rows, img_cols, all_patches_mixed_train_part1, all_patches_mixed_test_part1, pop_map_root, inference_patch_size, overlap, testlevels
from utils.constants import inference_patch_size as ips

import nvidia_smi
nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.args.probabilistic = False

        # set up experiment folder
        self.args.experiment_folder = os.path.join("/",os.path.join(*args.resume.split("/")[:-1]), "eval_outputs")
        self.experiment_folder = self.args.experiment_folder

        # seed before dataloader initialization
        seed_all(args.seed)

        # set up dataloaders
        self.dataloaders = self.get_dataloaders(self, args)
        
        # define architecture
        if args.model in model_dict:
            model_kwargs = get_model_kwargs(args, args.model)
            self.model = model_dict[args.model](**model_kwargs).cuda()
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        if args.model in ["BoostUNet"]:
            self.boosted = True
        else:
            self.boosted = False

        # number of params
        args.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Model", args.model, "; #Params:", args.pytorch_total_params)


        # wandb config
        wandb.init(project=args.wandb_project, dir=self.args.experiment_folder)
        wandb.config.update(self.args)

        # seed after initialization
        seed_all(args.seed+2)

        # initialize log dict
        self.info = { "epoch": 0,  "iter": 0,  "sampleitr": 0}

        # in case of checkpoint resume
        if args.resume is not None:
            self.resume(path=args.resume)


    def test_target(self, save=False, full=True):
        # Test on target domain
        self.model.eval()
        self.test_stats = defaultdict(float)
        # self.model.train()

        with torch.no_grad(): 
            self.target_test_stats = defaultdict(float)
            for testdataloader in self.dataloaders["test_target"]:
                if testdataloader.dataset.region in ["uga"]:
                    continue

                # inputialize the output map
                h, w = testdataloader.dataset.shape()
                output_map = torch.zeros((h, w), dtype=torch.float16)
                output_scale_map = torch.zeros((h, w), dtype=torch.float16)
                output_map_count = torch.zeros((h, w), dtype=torch.int8)

                if self.args.probabilistic:
                    output_map_var = torch.zeros((h, w), dtype=torch.float16)
                if self.boosted and full:
                    output_map_raw = torch.zeros((h, w), dtype=torch.float16)
                    if self.args.probabilistic:
                        output_map_var_raw = torch.zeros((h, w), dtype=torch.float16)

                for sample in tqdm(testdataloader, leave=True):
                    sample = to_cuda_inplace(sample)
                    sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput, segmentationinput=self.args.segmentationinput)

                    # get the valid coordinates
                    xl,yl = [val.item() for val in sample["img_coords"]]
                    mask = sample["mask"][0].bool()

                    # get the output with a forward pass
                    output = self.model(sample, padding=False)
                    output_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["popdensemap"][0][mask].cpu().to(torch.float16)
                    if self.args.probabilistic:
                        output_map_var[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["popvarmap"][0][mask].cpu().to(torch.float16)
                    if self.boosted and full:
                        output_map_raw[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["intermediate"]["popdensemap"][0][mask].cpu().to(torch.float16)
                        if self.args.probabilistic:
                            output_map_var_raw[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["intermediate"]["popvarmap"][0][mask].cpu().to(torch.float16) 

                    if "scale" in output.keys():
                        output_scale_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["scale"][0][mask].cpu().to(torch.float16)

                    output_map_count[xl:xl+ips, yl:yl+ips][mask.cpu()] += 1

                # average over the number of times each pixel was visited

                # mask out values that are not visited of visited exactly once
                div_mask = output_map_count > 1
                output_map[div_mask] = output_map[div_mask] / output_map_count[div_mask]
                if self.args.probabilistic:
                    output_map_var[div_mask] = output_map_var[div_mask] / output_map_count[div_mask]
                if self.boosted:
                    output_map_raw[div_mask] = output_map_raw[div_mask] / output_map_count[div_mask]
                    if self.args.probabilistic: 
                        output_map_var_raw[div_mask] = output_map_var_raw[div_mask] / output_map_count[div_mask]

                if "scale" in output.keys():
                    output_scale_map[div_mask] = output_scale_map[div_mask] / output_map_count[div_mask]
                
                # save maps
                if save:
                    # save the output map
                    testdataloader.dataset.save(output_map, self.experiment_folder)
                    if self.args.probabilistic:
                        testdataloader.dataset.save(output_map_var, self.experiment_folder, tag="VAR_{}".format(testdataloader.dataset.region))
                    if self.boosted and full:
                        testdataloader.dataset.save(output_map_raw, self.experiment_folder, tag="RAW_{}".format(testdataloader.dataset.region))
                        if self.args.probabilistic:
                            testdataloader.dataset.save(output_map_var_raw, self.experiment_folder, tag="VAR_RAW_{}".format(testdataloader.dataset.region))

                    if "scale" in output.keys():
                        testdataloader.dataset.save(output_scale_map, self.experiment_folder, tag="SCALE_{}".format(testdataloader.dataset.region))
                
                # convert populationmap to census
                for level in testlevels[testdataloader.dataset.region]:
                    # convert map to census
                    census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map, gpu_mode=True, level=level, details_to=os.path.join(self.experiment_folder, "{}_{}".format(testdataloader.dataset.region, level)))
                    this_metrics = get_test_metrics(census_pred, census_gt.float().cuda(), tag="MainCensus_{}_{}".format(testdataloader.dataset.region, level))
                    print(this_metrics)
                    self.target_test_stats = {**self.target_test_stats, **this_metrics}

                    # get the metrics for the clearly built up areas
                    built_up = census_gt>10
                    self.target_test_stats = {**self.target_test_stats,
                                              **get_test_metrics(census_pred[built_up], census_gt[built_up].float().cuda(), tag="MainCensusPos_{}_{}".format(testdataloader.dataset.region, level))}
                    
                    if self.boosted:
                        census_pred_raw, census_gt_raw = testdataloader.dataset.convert_popmap_to_census(output_map_raw, gpu_mode=True, level=level)
                        self.target_test_stats = {**self.target_test_stats,
                                                  **get_test_metrics(census_pred_raw, census_gt_raw.float().cuda(), tag="CensusRaw_{}_{}".format(testdataloader.dataset.region, level))}
                        built_up = census_gt_raw>10
                        self.target_test_stats = {**self.target_test_stats,
                                                  **get_test_metrics(census_pred_raw[built_up], census_gt_raw[built_up].float().cuda(), tag="CensusRawPos_{}_{}".format(testdataloader.dataset.region, level))}

                    # create scatterplot and upload to wandb
                    # print(self.target_test_stats)
                    scatterplot = scatter_plot3(census_pred.tolist(), census_gt.tolist(), log_scale=True)
                    if scatterplot is not None:
                        self.target_test_stats["Scatter/Scatter_{}_{}".format(testdataloader.dataset.region, level)] = wandb.Image(scatterplot)
                
                # adjust map (disaggregate) and recalculate everything
                print("-"*50)
                print("Adjusting map")
                output_map_adj = testdataloader.dataset.adjust_map_to_census(output_map)

                # save adjusted map
                if save:
                    testdataloader.dataset.save(output_map_adj, self.experiment_folder, tag="ADJ_{}".format(testdataloader.dataset.region))

                for level in testlevels[testdataloader.dataset.region]:
                    # convert map to census
                    census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map_adj, gpu_mode=True, level=level, details_to=os.path.join(self.experiment_folder, "{}_{}_adj".format(testdataloader.dataset.region, level)))
                    test_stats_adj = get_test_metrics(census_pred, census_gt.float().cuda(), tag="AdjCensus_{}_{}".format(testdataloader.dataset.region, level))
                    built_up = census_gt>10
                    test_stats_adj = {**test_stats_adj,
                                      **get_test_metrics(census_pred[built_up], census_gt[built_up].float().cuda(), tag="AdjCensusPos_{}_{}".format(testdataloader.dataset.region, level))}
                    
                    print(test_stats_adj)
                    self.target_test_stats = {**self.target_test_stats,
                                              **test_stats_adj}

                    scatterplot = scatter_plot3(census_pred.tolist(), census_gt.tolist(), log_scale=True)
                    if scatterplot is not None:
                        self.target_test_stats["Scatter/Scatter_{}_{}_adj".format(testdataloader.dataset.region, level)] = wandb.Image(scatterplot)
                    
            
            # save the target test stats
            wandb.log({**{k + '/targettest': v for k, v in self.target_test_stats.items()}, **self.info}, self.info["iter"])


    @staticmethod
    def get_dataloaders(self, args): 
        """
        Get dataloaders for the source and target domains
        Inputs:
            args: command line arguments
            force_recompute: if True, recompute the dataloader's and look out for new files even if the file list already exist
        Outputs:
            dataloaders: dictionary of dataloaders
        """

        input_defs = {'S1': args.Sentinel1, 'S2': args.Sentinel2, 'VIIRS': args.VIIRS, 'NIR': args.NIR}

        self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats_unified_2A.json'))
        for mkey in self.dataset_stats.keys():
            if isinstance(self.dataset_stats[mkey], dict):
                for key,val in self.dataset_stats[mkey].items():
                    self.dataset_stats[mkey][key] = torch.tensor(val)
            else:
                self.dataset_stats[mkey] = torch.tensor(val)

        # create the raw source dataset
        datasets = {
            "test_target": [ Population_Dataset_target(reg, patchsize=ips, overlap=overlap, sentinelbuildings=args.sentinelbuildings,
                                                       fourseasons=self.args.fourseasons, train_level=lvl, **input_defs)
                                for reg,lvl in zip(args.target_regions, args.train_level) ]
        }
        
        # create the dataloaders
        dataloaders =  {
            "test_target":  [DataLoader(datasets["test_target"], batch_size=1, num_workers=1, shuffle=False, drop_last=False)
                                for datasets["test_target"] in datasets["test_target"] ]
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
        self.info["epoch"] = checkpoint['epoch']
        self.info["iter"] = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = eval_parser.parse_args()
    print(eval_parser.format_values())

    trainer = Trainer(args)

    since = time.time() 
    trainer.test_target(save=True)
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
