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
# from utils.transform import Eu2Rwa
from tqdm import tqdm

from torchcontrib.optim import SWA

import itertools
import random
from sklearn import model_selection
import wandb

import pickle
import gc

# from arguments import train_parser
from arguments.train import parser as train_parser
# from data.So2Sat import PopulationDataset_Reg
from data.PopulationDataset_target import Population_Dataset_target, Population_Dataset_collate_fn
from utils.losses import get_loss, r2
from utils.metrics import get_test_metrics
from utils.utils import new_log, to_cuda, to_cuda_inplace, detach_tensors_in_dict, seed_all
# from utils.utils import new_log, to_cuda, to_cuda_inplace, detach_tensors_in_dict, seed_all, get_model_kwargs, model_dict
from model.get_model import get_model_kwargs, model_dict
from utils.utils import load_json, apply_transformations_and_normalize, apply_normalize
from utils.constants import config_path
from utils.scheduler import CustomLRScheduler

from utils.plot import plot_2dmatrix, plot_and_save, scatter_plot3
# from utils.utils import get_fnames_labs_reg, get_fnames_unlab_reg
# from utils.datasampler import LabeledUnlabeledSampler
from utils.constants import img_rows, img_cols, all_patches_mixed_train_part1, all_patches_mixed_test_part1, pop_map_root, testlevels, overlap
from utils.constants import inference_patch_size as ips
from utils.utils import Namespace


torch.autograd.set_detect_anomaly(True)

import nvidia_smi
nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args


        if args.loss in ["gaussian_nll", "log_gaussian_nll", "laplacian_nll", "log_laplacian_nll", "gaussian_aug_loss", "log_gaussian_aug_loss", "laplacian_aug_loss", "log_laplacian_aug_loss"]:
            self.args.probabilistic = True
        else:
            self.args.probabilistic = False

        # set up experiment folder
        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, "So2Sat"), args)
        self.args.experiment_folder = self.experiment_folder
        
        # seed everything
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

        # set up model
        seed_all(args.seed+1)
        
        # number of params
        args.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print("Model", args.model, "; #Params:", args.pytorch_total_params)
        args.num_effective_param = self.model.num_params
        print("Model", args.model, "; #Effective Params:", args.num_effective_param)


        # wandb config
        wandb.init(project=args.wandb_project, dir=self.experiment_folder)
        wandb.config.update(self.args)
        wandb.watch(self.model, log='all')  
        
        # seed after initialization
        seed_all(args.seed+2)

        # set up optimizer and scheduler
        if args.optimizer == "Adam":
            
            
            head_name = ['head.6.weight','head.6.bias']

            # Get all parameters except the head bias
            params_with_decay = [param for name, param in self.model.named_parameters() if name not in head_name and 'embedder' not in name]

            # check if the model has an embedder
            if hasattr(self.model, 'embedder'):
                # Get the positional embedding parameters
                params_positional = [param for name, param in self.model.embedder.named_parameters()]
            else:
                params_positional = []

            # Get the head bias parameter, only bias, if available
            params_without_decay = [param for name, param in self.model.named_parameters() if name in head_name and 'embedder' not in name]

            # self.optimizer = optim.Adam([
            #         {'params': params_with_decay, 'weight_decay': args.weightdecay, "lr": args.learning_rate}, # Apply weight decay here
            #         {'params': params_positional, 'weight_decay': args.weightdecay_pos, "lr": args.learning_rate}, # Apply weight decay here
            #         {'params': params_without_decay, 'weight_decay': 0.0, "lr": args.learning_rate/10}, # No weight decay
            #     ]
            #     , lr=args.learning_rate)
            
            self.optimizer = optim.Adam([
                    {'params': params_with_decay, 'weight_decay': args.weightdecay}, # Apply weight decay here
                    {'params': params_positional, 'weight_decay': args.weightdecay_pos}, # Apply weight decay here
                    {'params': params_without_decay, 'weight_decay': 0.0}, # No weight decay
                ]
                , lr=args.learning_rate)
            
            if args.resume_extractor is not None:
                self.optimizer = optim.Adam([ {'params': self.model.unetmodel.parameters(), 'weight_decay': args.weightdecay}]  , lr=args.learning_rate)
                
        elif args.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
            # self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weightdecay)


        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        # self.scheduler = CustomLRScheduler(self.optimizer, drop_epochs=[3, 5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 110, 150, ], gamma=0.75)
        
        # set up info
        self.info = { "epoch": 0,  "iter": 0,  "sampleitr": 0}
        self.info["alpha"], self.info["beta"] = 0.0, 0.0
        self.train_stats, self.val_stats = defaultdict(lambda: np.nan), defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        # in case of checkpoint resume
        if args.resume is not None:
            self.resume(path=args.resume)
        if args.resume_extractor is not None:
            self.resume(path=args.resume_extractor, load_optimizer=False)

    def train(self):
        """
        Main training loop
        """
        with tqdm(range(self.info["epoch"], self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                # if self.args.supmode=="weaksup" and self.args.weak_validation:
                #     self.validate_weak()
                
                # self.test_target(save=True)


                self.train_epoch(tnr)
                torch.cuda.empty_cache()

                # in domain validation
                # if (self.info["epoch"] + 1) % self.args.val_every_n_epochs == 0:
                #     self.validate()
                #     torch.cuda.empty_cache()

                    # TODO weak validation
                if (self.info["epoch"] + 1) % self.args.val_every_n_epochs == 0:
                    if self.args.supmode=="weaksup" and self.args.weak_validation:
                        self.validate_weak()
                        torch.cuda.empty_cache()

                    # self.validate()
                    # torch.cuda.empty_cache()

                    # if self.args.supmode=="weaksup":
                    #     self.validate_weak()
                    #     torch.cuda.empty_cache()
                
                if (self.info["epoch"] + 1) % (1*self.args.val_every_n_epochs) == 0:
                    self.test_target(save=True)
                    torch.cuda.empty_cache()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')
                
                # logging and scheduler step
                if self.args.lr_gamma != 1.0: 
                    self.scheduler.step()
                    wandb.log({**{'log_lr': np.log10(self.scheduler.get_last_lr())}, **self.info}, self.info["iter"])
                
                self.info["epoch"] += 1

    def train_epoch(self, tnr=None):
        """
        Train for one epoch
        """
        train_stats = defaultdict(float)

        # set model to train mode
        self.model.train()

        # get GPU memory usage
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        train_stats["gpu_used"] = info.used / 1e9 # in GB 

        # check if we are in unsupervised or supervised mode and adjust dataloader accordingly
        dataloader = self.dataloaders['train'] 
        self.optimizer.zero_grad()

        with tqdm(dataloader, leave=False, total=len(dataloader)) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)

            # iterate over samples of one epoch
            for i, sample in enumerate(inner_tnr):
                # self.optimizer.zero_grad()
                optim_loss = 0.0
                loss_dict_weak = {}
                loss_dict_raw = {}
                

                #  check if sample is weakly target supervised or source supervised 
                if self.args.supmode=="weaksup":
                    
                    # forward pass and loss computation
                    sample_weak = to_cuda_inplace(sample) 
                    sample_weak = apply_transformations_and_normalize(sample_weak, self.data_transform, self.dataset_stats, buildinginput=self.args.buildinginput, segmentationinput=self.args.segmentationinput)
                    
                    # check if the input is to large 
                    num_pix = sample_weak["input"].shape[0]*sample_weak["input"].shape[2]*sample_weak["input"].shape[3]
                    # limit1, limit2, limit3 = 10000000, 12500000, 15000000
                    limit1, limit2, limit3 =    4000000,  500000, 12000000
                    # limit1, limit2, limit3 =    2000000,  300000, 10000000

                    encoder_no_grad, unet_no_grad = False, False 
                    if num_pix > limit1:
                        encoder_no_grad, unet_no_grad = True, False
                        if num_pix > limit2:
                            encoder_no_grad, unet_no_grad = True, True 
                            if num_pix > limit3:
                                print("Input to large for encoder and unet")
                                continue

                    output_weak = self.model(sample_weak, train=True, alpha=0., return_features=False, padding=False,
                                             encoder_no_grad=encoder_no_grad, unet_no_grad=unet_no_grad, sparse=True)

                    # merge augmented samples
                    if self.args.weak_merge_aug:
                        output_weak["popcount"] = output_weak["popcount"].sum(dim=0, keepdim=True)
                        if "popvar" in output_weak:
                            output_weak["popvar"] = output_weak["popvar"].sum(dim=0, keepdim=True) 
                        sample_weak["y"] = sample_weak["y"].sum(dim=0, keepdim=True)
                        sample_weak["source"] = sample_weak["source"][0]
                        if self.boosted:
                            output_weak["intermediate"]["popcount"] = output_weak["intermediate"]["popcount"].sum(dim=0, keepdim=True)
                            if "popvar" in output_weak["intermediate"]:
                                output_weak["intermediate"]["popvar"] = output_weak["intermediate"]["popvar"].sum(dim=0, keepdim=True)

                    # compute loss
                    loss_weak, loss_dict_weak = get_loss(
                        output_weak, sample_weak, scale=output_weak["scale"], empty_scale=output_weak["empty_scale"], loss=args.loss, lam=args.lam, merge_aug=args.merge_aug,
                        scale_regularization=args.scale_regularization, scale_regularizationL2=args.scale_regularizationL2, emptyscale_regularizationL2=args.emptyscale_regularizationL2,
                        tag="weak")
                    
                    # Detach tensors
                    loss_dict_weak = detach_tensors_in_dict(loss_dict_weak)
                    
                    if self.boosted:
                        boosted_loss = [el.replace("gaussian", "l1") if el in ["gaussian_nll", "log_gaussian_nll", "gaussian_aug_loss", "log_gaussian_aug_loss"] else el for el in args.loss]
                        boosted_loss = [el.replace("laplace", "l1") if el in ["laplacian_nll", "log_laplacian_nll", "laplace_aug_loss", "log_laplace_aug_loss"] else el for el in boosted_loss]
                        loss_weak_raw, loss_weak_dict_raw = get_loss(
                            output_weak["intermediate"], sample_weak, scale=output_weak["intermediate"]["scale"], empty_scale=output_weak["intermediate"]["empty_scale"], loss=boosted_loss,
                            lam=args.lam, merge_aug=args.merge_aug, scale_regularization=args.scale_regularization, scale_regularizationL2=args.scale_regularizationL2, emptyscale_regularizationL2=args.emptyscale_regularizationL2,
                            tag="train_weak_intermediate")
                        
                        loss_weak_dict_raw = detach_tensors_in_dict(loss_weak_dict_raw)
                        loss_dict_weak = {**loss_dict_weak, **loss_weak_dict_raw}
                        loss_weak += loss_weak_raw * self.args.lam_raw

                    # update loss
                    optim_loss += loss_weak * self.args.lam_weak #* self.info["beta"]
                else:
                    output_weak = None

                loss_dict = {}
                loss_dict_raw = {}
                output = None

                # Detach tensors
                loss_dict = detach_tensors_in_dict(loss_dict)

                # accumulate statistics of all dicts
                for key in loss_dict:
                    train_stats[key] += loss_dict[key].cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                for key in loss_dict_weak:
                    train_stats[key] += loss_dict_weak[key].cpu().item() if torch.is_tensor(loss_dict_weak[key]) else loss_dict_weak[key]
                for key in loss_dict_raw:
                    train_stats[key] += loss_dict_raw[key].cpu().item() if torch.is_tensor(loss_dict_raw[key]) else loss_dict_raw[key]
                train_stats["log_count"] += 1

                # detect NaN loss 
                # backprop and stuff
                if torch.isnan(optim_loss):
                    raise Exception("detected NaN loss..")
                
                if self.info["epoch"] > 0 or not self.args.no_opt:
                    # backprop
                    optim_loss.backward()

                    # gradient clipping
                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
                    
                    if (i + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    optim_loss = optim_loss.detach()
                    if output is not None:  
                        output = detach_tensors_in_dict(output)
                        del output
                    if output_weak is not None:
                        output_weak = detach_tensors_in_dict(output_weak)
                        del output_weak
                    torch.cuda.empty_cache()
                    del sample 
                    gc.collect()

                # update info
                self.info["iter"] += 1
                self.info["sampleitr"] += self.args.batch_size
                
                # logging and stuff
                if (i+1) % self.args.val_every_i_steps == 0:
                    if self.args.supmode=="weaksup" and self.args.weak_validation:
                        self.log_train(train_stats)
                        self.validate_weak()
                        self.model.train()

                # logging and stuff
                if (i+1) % self.args.test_every_i_steps == 0:
                    self.log_train(train_stats)
                    self.test_target(save=True)
                    self.model.train()

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.log_train(train_stats,(inner_tnr, tnr))
                    train_stats = defaultdict(float)
    
    def log_train(self, train_stats, tqdmstuff=None):
        train_stats = {k: v / train_stats["log_count"] for k, v in train_stats.items()}

        # print logs to console via tqdm
        if tqdmstuff is not None:
            inner_tnr, tnr = tqdmstuff
            inner_tnr.set_postfix(training_loss=train_stats['optimization_loss'])
            if tnr is not None:
                tnr.set_postfix(training_loss=train_stats['optimization_loss'],
                                validation_loss=self.val_stats['optimization_loss'],
                                best_validation_loss=self.best_optimization_loss)

        # upload logs to wandb
        wandb.log({**{k + '/train': v for k, v in train_stats.items()}, **self.info}, self.info["iter"])
        

    def validate_weak(self):
        self.valweak_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for valdataloader in self.dataloaders["weak_target_val"]:
                pred, gt = [], []
                for i,sample in enumerate(tqdm(valdataloader, leave=False)):
                    sample = to_cuda_inplace(sample)
                    sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput, segmentationinput=self.args.segmentationinput)

                    output = self.model(sample, padding=False)

                    # Colellect predictions and samples
                    pred.append(output["popcount"]); gt.append(sample["y"])

                # compute metrics
                pred = torch.cat(pred); gt = torch.cat(gt)
                self.valweak_stats = { **self.valweak_stats,
                                       **get_test_metrics(pred, gt.float().cuda(), tag="MainCensus_{}_{}".format(valdataloader.dataset.region, self.args.train_level))  }

            wandb.log({**{k + '/val': v for k, v in self.valweak_stats.items()}, **self.info}, self.info["iter"])

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

                for sample in tqdm(testdataloader, leave=False):
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
                    census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map, gpu_mode=True, level=level)
                    self.target_test_stats = {**self.target_test_stats,
                                              **get_test_metrics(census_pred, census_gt.float().cuda(), tag="MainCensus_{}_{}".format(testdataloader.dataset.region, level))}
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
                    scatterplot = scatter_plot3(census_pred.tolist(), census_gt.tolist(), log_scale=True)
                    if scatterplot is not None:
                        self.target_test_stats["Scatter/Scatter_{}_{}".format(testdataloader.dataset.region, level)] = wandb.Image(scatterplot)

            wandb.log({**{k + '/targettest': v for k, v in self.target_test_stats.items()}, **self.info}, self.info["iter"])
        

    @staticmethod
    def get_dataloaders(self, args): 
        """
        Get dataloaders for the source and target domains
        Inputs:
            args: command line arguments 
        Outputs:
            dataloaders: dictionary of dataloaders
        """

        input_defs = {'S1': args.Sentinel1, 'S2': args.Sentinel2, 'VIIRS': args.VIIRS, 'NIR': args.NIR}
        params = {'dim': (img_rows, img_cols), "satmode": args.satmode, 'in_memory': args.in_memory, **input_defs}
        self.data_transform = {}
        if args.full_aug:
            self.data_transform["general"] = transforms.Compose([
                # AddGaussianNoise(std=0.04, p=0.75),
                RandomVerticalFlip(p=0.5, allsame=args.supmode=="weaksup"),
                RandomHorizontalFlip(p=0.5, allsame=args.supmode=="weaksup"),
                RandomRotationTransform(angles=[90, 180, 270], p=0.75),
            ])
            S2augs = [
                RandomBrightness(p=0.9, beta_limit=(0.666, 1.5)),
                RandomGamma(p=0.9, gamma_limit=(0.6666, 1.5)),
            ]
        else: 
            self.data_transform["general"] = transforms.Compose([  ])
            S2augs = []

        # collect all transformations
        self.data_transform["S2"] = OwnCompose(S2augs)
        self.data_transform["S1"] = transforms.Compose([ ])
        
        # load normalization stats
        self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats_unified_2A.json'))
        for mkey in self.dataset_stats.keys():
            if isinstance(self.dataset_stats[mkey], dict):
                for key,val in self.dataset_stats[mkey].items():
                    self.dataset_stats[mkey][key] = torch.tensor(val)
            else:
                self.dataset_stats[mkey] = torch.tensor(val)

        datasets = {
            "test_target": [ Population_Dataset_target( reg, patchsize=ips, overlap=overlap, sentinelbuildings=args.sentinelbuildings, **input_defs) \
                                for reg in args.target_regions ]
        }

        # create the dataloaders
        dataloaders =  {
            "test_target":  [DataLoader(datasets["test_target"], batch_size=1, num_workers=1, shuffle=False, drop_last=False) \
                                for datasets["test_target"] in datasets["test_target"] ]
        }
        
        # add weakly supervised samples of the target domain to the trainind_dataset
        # create the weakly supervised dataset stack them into a single dataset and dataloader
        if args.supmode=="weaksup":
            if args.gradientaccumulation:
                weak_loader_batchsize = 1
                self.accumulation_steps = args.weak_batch_size
            else:
                weak_loader_batchsize = args.weak_batch_size
                self.accumulation_steps = 1
                
            weak_datasets = []
            # for reg in args.target_regions_train:
            for reg, lvl in zip(args.target_regions_train, args.train_level):
                splitmode = 'train' if self.args.weak_validation else 'all'
                weak_datasets.append( Population_Dataset_target(reg, mode="weaksup", split=splitmode, patchsize=None, overlap=None, max_samples=args.max_weak_samples,
                                                                fourseasons=args.random_season, transform=None, sentinelbuildings=args.sentinelbuildings, 
                                                                ascfill=True, train_level=lvl, max_pix=self.args.max_weak_pix, ascAug=args.ascAug, **input_defs)  )
            dataloaders["weak_target_dataset"] = ConcatDataset(weak_datasets)
            dataloaders["train"] = DataLoader(dataloaders["weak_target_dataset"], batch_size=weak_loader_batchsize, num_workers=1, shuffle=True, collate_fn=Population_Dataset_collate_fn, drop_last=True)
            
            weak_datasets_val = []
            if self.args.weak_validation:
                # for reg in list(set(args.target_regions) | set(args.target_regions_train)):
                for reg, lvl in zip(args.target_regions_train, args.train_level):
                    weak_datasets_val.append(Population_Dataset_target(reg, mode="weaksup", split="val", patchsize=None, overlap=None, max_samples=args.max_weak_samples,
                                                                    fourseasons=args.random_season, transform=None, sentinelbuildings=args.sentinelbuildings, 
                                                                    ascfill=True, train_level=lvl, max_pix=self.args.max_weak_pix, **input_defs) )
                dataloaders["weak_target_val"] = [ DataLoader(weak_datasets_val[i], batch_size=self.args.weak_val_batch_size, num_workers=1, shuffle=False, collate_fn=Population_Dataset_collate_fn, drop_last=True) for i in range(len(args.target_regions_train)) ]

        return dataloaders
   

    def save_model(self, prefix=''):
        """
        Input:
            prefix: string to prepend to the filename
        """
        torch.save({
            'model': self.model.state_dict(),
            'epoch': self.info["epoch"] + 1,
            'iter': self.info["iter"],
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

        # if there is a self.model.unet, save the unet as well
        if hasattr(self.model, "unetmodel"):
            torch.save({
                'model': self.model.unetmodel.state_dict(),
            }, os.path.join(self.experiment_folder, f'{prefix}_unetmodel.pth'))

        # if there is a self.model.head, save the head as well
        if hasattr(self.model, "head"):
            torch.save({
                'model': self.model.head.state_dict(),
            }, os.path.join(self.experiment_folder, f'{prefix}_head.pth'))

        if hasattr(self.model, "embedder"):
            torch.save({
                'model': self.model.embedder.state_dict(),
            }, os.path.join(self.experiment_folder, f'{prefix}_embedder.pth'))


    def resume(self, path, load_optimizer=True):
        """
        Input:
            path: path to the checkpoint
        """
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        # load checkpoint
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.info["epoch"] = checkpoint['epoch']
        self.info["iter"] = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = train_parser.parse_args()
    print(train_parser.format_values())

    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
