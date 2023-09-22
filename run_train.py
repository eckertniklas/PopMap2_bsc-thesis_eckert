import os
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
from torch import is_tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Normalize
from torchvision import transforms
from utils.transform import OwnCompose
from utils.transform import RandomRotationTransform, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomHorizontalVerticalFlip, RandomBrightness, RandomGamma, HazeAdditionModule, AddGaussianNoise, AddGaussianNoiseWithCorrelation
# from utils.transform import Eu2Rwa
from tqdm import tqdm
 
from torch.cuda.amp import autocast, GradScaler

import wandb
 
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
from utils.utils import Namespace, NumberList

import rasterio
from rasterio.windows import Window
from shutil import copyfile

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
            params_with_decay = [param for name, param in self.model.named_parameters() if name not in head_name and 'embedder' not in name and 'unetmodel' not in name]

            # check if the model has an embedder
            if hasattr(self.model, 'embedder'):
                # Get the positional embedding parameters
                params_positional = [param for name, param in self.model.embedder.named_parameters()]
            else:
                params_positional = []

            params_unet_only = [param for name, param in self.model.named_parameters() if name not in head_name and 'embedder' not in name and 'unetmodel' in name]

            # Get the head bias parameter, only bias, if available
            params_without_decay = [param for name, param in self.model.named_parameters() if name in head_name and 'embedder' not in name and 'unetmodel' not in name]

            self.optimizer = optim.Adam([
                    {'params': params_with_decay, 'weight_decay': args.weightdecay}, # Apply weight decay here
                    {'params': params_positional, 'weight_decay': args.weightdecay_pos}, # Apply weight decay here
                    {'params': params_unet_only, 'weight_decay': args.weightdecay_unet}, # No weight decay
                    {'params': params_without_decay, 'weight_decay': 0.0}, # No weight decay
                ]
                , lr=args.learning_rate)
            
            if args.resume_extractor is not None:
                self.optimizer = optim.Adam([ {'params': self.model.unetmodel.parameters(), 'weight_decay': args.weightdecay}]  , lr=args.learning_rate)
                
        elif args.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)

        if args.half:
            self.scaler = GradScaler()

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
        self.pred_buffer = NumberList()
        self.target_buffer = NumberList()

        with tqdm(range(self.info["epoch"], self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:               
                # self.test_target_large(save=True)
                # self.test_target(save=True)

                self.train_epoch(tnr)
                torch.cuda.empty_cache()

                if self.args.save_model in ['last', 'both']:
                    self.save_model('last')


                # weak validation
                if (self.info["epoch"] + 1) % self.args.val_every_n_epochs == 0:
                    if self.args.supmode=="weaksup" and self.args.weak_validation:
                        self.validate_weak()
                        torch.cuda.empty_cache()

                    # self.validate()
                    # torch.cuda.empty_cache()
                
                if (self.info["epoch"] + 1) % (self.args.val_every_n_epochs) == 0:
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
                    # sample_weak = to_cuda_inplace(sample, self.args.half, spare=["y", "source"]) 
                    sample_weak = to_cuda_inplace(sample) 
                    sample_weak = apply_transformations_and_normalize(sample_weak, self.data_transform, self.dataset_stats, buildinginput=self.args.buildinginput,
                                                                      segmentationinput=self.args.segmentationinput, empty_eps=self.args.empty_eps)
                    
                    # check if the input is to large
                    if sample_weak["input"] is not None:
                        num_pix = sample_weak["input"].shape[0]*sample_weak["input"].shape[2]*sample_weak["input"].shape[3]
                    else:
                        num_pix = 0

                    # limit1, limit2, limit3 = 10000000, 12500000, 15000000
                    # limit1, limit2, limit3 = 7000000,  1000000, 15000000
                    # limit1, limit2, limit3 = 14000000,  18000000, 22000000
                    # limit1, limit2, limit3 = 22000000,  44000000, 44000000
                    # limit1, limit2, limit3 = 16000000,  2500000, 2500000
                    # limit1, limit2, limit3 =    4000000,  500000, 12000000

                    encoder_no_grad, unet_no_grad = False, False 
                    if num_pix > self.args.limit1:
                        encoder_no_grad, unet_no_grad = True, False
                        print("Feezing encoder")
                        if num_pix > self.args.limit2:
                            encoder_no_grad, unet_no_grad = True, True 
                            print("Feezing decoder")
                            if num_pix > self.args.limit3:
                                print("Input to large for encoder and unet. No forward pass.")
                                continue
                    
                    if self.args.half:
                        with autocast():
                            output_weak = self.model(sample_weak, train=True, alpha=0., return_features=False, padding=False,
                                                encoder_no_grad=encoder_no_grad, unet_no_grad=unet_no_grad,
                                                # sparse=self.args.empty_eps>0.0
                                                sparse=True
                                                )
                    else:
                        output_weak = self.model(sample_weak, train=True, alpha=0., return_features=False, padding=False,
                                                encoder_no_grad=encoder_no_grad, unet_no_grad=unet_no_grad,
                                                # sparse=self.args.empty_eps>0.0
                                                sparse=True
                                                )

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
                    if self.args.half:
                        with autocast():
                            loss_weak, loss_dict_weak = get_loss(
                                output_weak, sample_weak, scale=output_weak["scale"], empty_scale=output_weak["empty_scale"], loss=args.loss, lam=args.lam, merge_aug=args.merge_aug,
                                scale_regularization=args.scale_regularization, scale_regularizationL2=args.scale_regularizationL2, emptyscale_regularizationL2=args.emptyscale_regularizationL2,
                                output_regularization=args.output_regularization,
                                tag="weak")
                    else:
                        loss_weak, loss_dict_weak = get_loss(
                            output_weak, sample_weak, scale=output_weak["scale"], empty_scale=output_weak["empty_scale"], loss=args.loss, lam=args.lam, merge_aug=args.merge_aug,
                            scale_regularization=args.scale_regularization, scale_regularizationL2=args.scale_regularizationL2, emptyscale_regularizationL2=args.emptyscale_regularizationL2,
                            output_regularization=args.output_regularization,
                            tag="weak")
                    
                    # Detach tensors
                    loss_dict_weak = detach_tensors_in_dict(loss_dict_weak)
                    
                    if self.boosted:
                        boosted_loss = [el.replace("gaussian", "l1") if el in ["gaussian_nll", "log_gaussian_nll", "gaussian_aug_loss", "log_gaussian_aug_loss"] else el for el in args.loss]
                        boosted_loss = [el.replace("laplace", "l1") if el in ["laplacian_nll", "log_laplacian_nll", "laplace_aug_loss", "log_laplace_aug_loss"] else el for el in boosted_loss]
                        loss_weak_raw, loss_weak_dict_raw = get_loss(
                            output_weak["intermediate"], sample_weak, scale=output_weak["intermediate"]["scale"], empty_scale=output_weak["intermediate"]["empty_scale"], loss=boosted_loss,
                            lam=args.lam, merge_aug=args.merge_aug, scale_regularization=args.scale_regularization, scale_regularizationL2=args.scale_regularizationL2, emptyscale_regularizationL2=args.emptyscale_regularizationL2,
                            output_regularization=args.output_regularization,
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

                # collect buffer
                self.pred_buffer.add(output_weak["popcount"].cpu().detach())
                self.target_buffer.add(sample_weak["y"].cpu().detach())
                                
                # detect NaN loss 
                if torch.isnan(optim_loss):
                    raise Exception("detected NaN loss..")
                if torch.isinf(optim_loss):
                    raise Exception("detected Inf loss..")
                
                # backprop
                if self.info["epoch"] > 0 or not self.args.no_opt:
                    if self.args.half:
                        self.scaler.scale(optim_loss).backward()
                        # if torch.isnan(output_weak["scale"].grad).any():
                        #     print("NaN values detected in the gradient of scale.")
                    else:
                        optim_loss.backward()

                    # gradient clipping
                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
                    
                    if (i + 1) % self.accumulation_steps == 0:
                        if self.args.half:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    optim_loss = optim_loss.detach()
                    if output_weak is not None:
                        output_weak = detach_tensors_in_dict(output_weak)
                        del output_weak
                    torch.cuda.empty_cache()
                    del sample 
                    gc.collect()

                # update info
                self.info["iter"] += 1 
                self.info["sampleitr"] += self.args.weak_batch_size
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
        train_stats["Population_weak/r2"] = r2(torch.tensor(self.pred_buffer.get()),torch.tensor(self.target_buffer.get()))

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
                    sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput,
                                                                 segmentationinput=self.args.segmentationinput, empty_eps=self.args.empty_eps)

                    output = self.model(sample, padding=False)

                    # Colellect predictions and samples
                    pred.append(output["popcount"]); gt.append(sample["y"])

                # compute metrics
                pred = torch.cat(pred); gt = torch.cat(gt)
                self.valweak_stats = { **self.valweak_stats,
                                       **get_test_metrics(pred, gt.float().cuda(), tag="MainCensus_{}_{}".format(valdataloader.dataset.region, self.args.train_level))  }

            wandb.log({**{k + '/val': v for k, v in self.valweak_stats.items()}, **self.info}, self.info["iter"])

    def test_target(self, save=False, full=True):

        # if afganistan or uganda is in the test set, we need to use the large test function
        if any([el in self.args.target_regions for el in ["afg", "uga"]]):
            self.test_target_large(save=save, full=full)
            return

        # Test on target domain
        self.model.eval()
        self.test_stats = defaultdict(float)

        with torch.no_grad(): 
            self.target_test_stats = defaultdict(float)
            for testdataloader in self.dataloaders["test_target"]:

                # inputialize the output map
                h, w = testdataloader.dataset.shape()
                output_map = torch.zeros((h, w), dtype=torch.float16)
                output_scale_map = torch.zeros((h, w), dtype=torch.float16)
                output_map_count = torch.zeros((h, w), dtype=torch.int8)

                for sample in tqdm(testdataloader, leave=False):
                    sample = to_cuda_inplace(sample)
                    sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput,
                                                                 segmentationinput=self.args.segmentationinput, empty_eps=self.args.empty_eps)

                    # get the valid coordinates
                    xl,yl = [val.item() for val in sample["img_coords"]]
                    mask = sample["mask"][0].bool()

                    # get the output with a forward pass
                    output = self.model(sample, padding=False)
                    output_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["popdensemap"][0][mask].cpu().to(torch.float16)
                    if "scale" in output.keys() and output["scale"] is not None:
                        output_scale_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["scale"][0][mask].cpu().to(torch.float16)

                    output_map_count[xl:xl+ips, yl:yl+ips][mask.cpu()] += 1

                # average over the number of times each pixel was visited
                # mask out values that are not visited of visited exactly once
                div_mask = output_map_count > 1
                output_map[div_mask] = output_map[div_mask] / output_map_count[div_mask]

                if "scale" in output.keys():
                    output_scale_map[div_mask] = output_scale_map[div_mask] / output_map_count[div_mask]
                
                # save maps
                if save:
                    # save the output map
                    testdataloader.dataset.save(output_map, self.experiment_folder)
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
                    
                    # create scatterplot and upload to wandb
                    scatterplot = scatter_plot3(census_pred.tolist(), census_gt.tolist(), log_scale=True)
                    if scatterplot is not None:
                        self.target_test_stats["Scatter/Scatter_{}_{}".format(testdataloader.dataset.region, level)] = wandb.Image(scatterplot)

            wandb.log({**{k + '/targettest': v for k, v in self.target_test_stats.items()}, **self.info}, self.info["iter"])

            del output_map, output_map_count, output_scale_map
        

    def test_target_large(self, save=False, full=True):
        # Test on target domain
        self.model.eval()
        self.test_stats = defaultdict(float)

        with torch.no_grad(): 
            self.target_test_stats = defaultdict(float)
            for testdataloader in self.dataloaders["test_target"]:

                # inputialize the output map
                chunk_size = 4096  # or any other reasonable value
                tmp_output_map_file = os.path.join(self.experiment_folder, 'tmp_output_map.tif')
                tmp_output_map_count_file = os.path.join(self.experiment_folder, 'tmp_output_map_count.tif')
                tmp_output_map_scale_file = os.path.join(self.experiment_folder, 'tmp_output_map_scale.tif')
                metadata1 = testdataloader.dataset._meta
                metadata1.update({'compress': 'PACKBITS', 'dtype': 'float32'})

                # Initialize temporary raster files and write zeros to them in chunks
                with rasterio.open(tmp_output_map_file, 'w', **metadata1) as tmp_dst:
                    
                    # Create an array of zeros with the shape of the chunk
                    zeros_chunk = np.zeros((chunk_size, chunk_size), dtype=metadata1['dtype'])
                    
                    # Chunked writing of zeros to the raster files
                    for i in tqdm(range(0, metadata1['height'], chunk_size)):
                        for j in tqdm(range(0, metadata1['width'], chunk_size), leave=False, disable=True):
                            # Adjust the shape of the chunk for edge cases
                            if i + chunk_size > metadata1['height'] or j + chunk_size > metadata1['width']:
                                current_zeros_chunk = np.zeros((min(chunk_size, metadata1['height'] - i), 
                                                        min(chunk_size, metadata1['width'] - j)), 
                                                    dtype=metadata1['dtype'])
                            else:
                                current_zeros_chunk = zeros_chunk
                            
                            window = Window(j, i, current_zeros_chunk.shape[1], current_zeros_chunk.shape[0])
                            tmp_dst.write(current_zeros_chunk, 1, window=window) 

                # Copy the initialized file to create the second file
                copyfile(tmp_output_map_file, tmp_output_map_count_file)
                copyfile(tmp_output_map_file, tmp_output_map_scale_file)


                # # Initialize temporary raster files
                with rasterio.open(tmp_output_map_file, 'r+', **metadata1) as tmp_dst:
                    with rasterio.open(tmp_output_map_count_file, 'r+', **metadata1) as tmp_count_dst:
                        with rasterio.open(tmp_output_map_scale_file, 'r+', **metadata1) as tmp_scale_dst:
                            # Iterate over the chunks of the testdataloader
                            for i, sample in tqdm(enumerate(testdataloader), leave=False, total=len(testdataloader)):
                                sample = to_cuda_inplace(sample)
                                sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput,
                                                                            segmentationinput=self.args.segmentationinput, empty_eps=self.args.empty_eps)

                                # get the valid coordinates
                                xl,yl = [val.item() for val in sample["img_coords"]]
                                mask = sample["mask"][0].bool()

                                # get the output with a forward pass
                                output = self.model(sample, padding=False)
                                
                                # Save current predictions to temporary file
                                xl, yl, xu, yu = xl, yl, xl+ips, yl+ips
                                window = Window(yl, xl, yu-yl, xu-xl)

                                # Read existing values, sum new values (accounting for mask), and write back
                                existing_values = tmp_dst.read(1, window=window).astype(np.float32)
                                existing_values[mask.cpu()] += output["popdensemap"][0][mask].cpu().numpy().astype(np.float32) # might want to perform this operation on the GPU
                                tmp_dst.write(existing_values, 1, window=window)

                                if "scale" in output.keys():
                                    existing_values_scale = tmp_scale_dst.read(1, window=window).astype(np.float32)
                                    existing_values_scale[mask.cpu()] += output["scale"][0][mask].cpu().numpy().astype(np.float32) # might want to perform this operation on the GPU
                                    tmp_scale_dst.write(existing_values_scale, 1, window=window)

                                # Read existing values, sum new values (accounting for mask), and write back for the inference count tracker
                                output_map_count = tmp_count_dst.read(1, window=window).astype(np.int32) 
                                output_map_count[mask.cpu().numpy()] += 1
                                tmp_count_dst.write(output_map_count, 1, window=window)

                                # if i == 400:
                                #     break


                # save predictions to file 
                metadata = testdataloader.dataset.metadata()
                metadata.update({"count": 1,
                                "dtype": "float32",
                                "compress": "lzw" })
                                # "compress": "PACKBITS" })
                # average predictions
                gpu_mode = False
                # gpu_mode = True
                reg = testdataloader.dataset.region
                outputmap_file = os.path.join(self.experiment_folder, '{}_predictions.tif'.format(reg))
                outputmap_scale = os.path.join(self.experiment_folder, '{}_predictionsSCALE.tif'.format(reg))
                # Read the temporary maps in chunks, average and write to the final output map
                with rasterio.open(tmp_output_map_file, 'r') as tmp_src, \
                    rasterio.open(tmp_output_map_count_file, 'r') as tmp_count_src, \
                    rasterio.open(tmp_output_map_scale_file, 'r') as tmp_scale_src, \
                    rasterio.open(outputmap_file, 'w', **metadata) as dst, \
                    rasterio.open(outputmap_scale, 'w', **metadata) as dst_scale:

                    h,w = tmp_src.shape

                    for i in tqdm(range(0, h, chunk_size)):
                        for j in tqdm(range(0, w, chunk_size), leave=False, disable=False):
                            # Adjust the shape of the chunk for edge cases
                            chunk_height = min(chunk_size, h - i)
                            chunk_width = min(chunk_size, w - j)
                            
                            # Read chunks
                            window = Window(j, i, chunk_width, chunk_height)
                                
                            count_chunk = tmp_count_src.read(1, window=window)
                                
                            # Average the data chunk
                            if gpu_mode:
                                count_chunk = torch.tensor(count_chunk, dtype=torch.float32).cuda()
                                div_mask_chunk = count_chunk > 0
                                if div_mask_chunk.sum() > 0:
                                    data_chunk = tmp_src.read(1, window=window)

                                    if count_chunk.max()>1:
                                        # to gpu
                                        data_chunk = torch.tensor(data_chunk, dtype=torch.float32).cuda()
                                        data_chunk_scale = torch.tensor(tmp_scale_src.read(1, window=window), dtype=torch.float32).cuda()

                                        data_chunk[div_mask_chunk] /= count_chunk[div_mask_chunk]
                                        data_chunk_scale[div_mask_chunk] /= count_chunk[div_mask_chunk]

                                        # back to cpu numpy
                                        data_chunk = data_chunk.cpu().numpy()
                                        data_chunk_scale = data_chunk_scale.cpu().numpy()
                                    else:
                                        pass

                                    # Write the chunk to the final file
                                    dst.write((data_chunk), 1, window=window)
                                    dst_scale.write((data_chunk_scale), 1, window=window)

                            else:
                                div_mask_chunk = count_chunk > 0
                                if div_mask_chunk.sum() > 0:
                                    data_chunk = tmp_src.read(1, window=window)
                                    data_chunk_scale = tmp_scale_src.read(1, window=window)

                                    if count_chunk.max()>1:

                                        data_chunk[div_mask_chunk] = data_chunk[div_mask_chunk] / count_chunk[div_mask_chunk]
                                        data_chunk_scale[div_mask_chunk] = data_chunk_scale[div_mask_chunk] / count_chunk[div_mask_chunk]
                                    else:
                                        pass
                                    
                                    # Write the chunk to the final file
                                    data_chunk = (data_chunk*2**16).astype(np.int16)/2**16
                                    dst.write((data_chunk), 1, window=window)

                                    # round to make compression more efficient
                                    # data_chunk_scale = int(data_chunk_scale*2**16)/2**16
                                    data_chunk_scale = (data_chunk_scale*2**16).astype(np.int16)/2**16
                                    dst_scale.write((data_chunk_scale).astype(np.float32), 1, window=window)

            del output_map_count, existing_values, existing_values_scale
            os.remove(tmp_output_map_file)
            os.remove(tmp_output_map_count_file)

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
            general_transforms = [
                # AddGaussianNoise(std=0.04, p=0.75), 
                RandomVerticalFlip(p=0.5, allsame=args.supmode=="weaksup"),
                RandomHorizontalFlip(p=0.5, allsame=args.supmode=="weaksup"),
                RandomRotationTransform(angles=[90, 180, 270], p=0.75),
            ]
            if args.addgaussiannoise:
                general_transforms.append(AddGaussianNoiseWithCorrelation(std=1.0, p=0.75))

            self.data_transform["general"] = transforms.Compose(general_transforms)

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

        # get the target regions for testing
        # ascfill = True if reg in ["uga"] else False
        need_asc = ["uga"]
        datasets = {
            "test_target": [ Population_Dataset_target( reg, patchsize=ips, overlap=overlap, sentinelbuildings=args.sentinelbuildings, ascfill=reg in need_asc, **input_defs) \
                                for reg in args.target_regions ] }
        dataloaders =  {
            "test_target":  [DataLoader(datasets["test_target"], batch_size=1, num_workers=self.args.num_workers, shuffle=False, drop_last=False) \
                                for datasets["test_target"] in datasets["test_target"] ]  }
        
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
                                                                ascfill=reg in need_asc, train_level=lvl, max_pix=self.args.max_weak_pix, max_pix_box=self.args.max_pix_box, ascAug=args.ascAug, **input_defs)  )
            dataloaders["weak_target_dataset"] = ConcatDataset(weak_datasets)
            dataloaders["train"] = DataLoader(dataloaders["weak_target_dataset"], batch_size=weak_loader_batchsize, num_workers=self.args.num_workers, shuffle=True, collate_fn=Population_Dataset_collate_fn, drop_last=True)
            
            weak_datasets_val = []
            if self.args.weak_validation: 
                for reg, lvl in zip(args.target_regions_train, args.train_level):
                    weak_datasets_val.append(Population_Dataset_target(reg, mode="weaksup", split="val", patchsize=None, overlap=None, max_samples=args.max_weak_samples,
                                                                    fourseasons=args.random_season, transform=None, sentinelbuildings=args.sentinelbuildings, 
                                                                    ascfill=reg in need_asc, train_level=lvl, max_pix=self.args.max_weak_pix, max_pix_box=self.args.max_pix_box, **input_defs) )
                dataloaders["weak_target_val"] = [ DataLoader(weak_datasets_val[i], batch_size=self.args.weak_val_batch_size, num_workers=self.args.num_workers, shuffle=False, collate_fn=Population_Dataset_collate_fn, drop_last=True)
                                                  for i in range(len(args.target_regions_train)) ]

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
            # if self.model.unetmodel is of type torch module
            if isinstance(self.model.unetmodel, torch.nn.Module):
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
