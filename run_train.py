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
from utils.transform import RandomRotationTransform, RandomHorizontalFlip, RandomVerticalFlip, RandomHorizontalVerticalFlip, RandomBrightness, RandomGamma, HazeAdditionModule, AddGaussianNoise
from tqdm import tqdm

import itertools
import random
from sklearn import model_selection
import wandb

import gc

from arguments import train_parser
from data.So2Sat import PopulationDataset_Reg
from data.PopulationDataset_target import Population_Dataset_target, Population_Dataset_collate_fn
from utils.losses import get_loss, r2
from utils.metrics import get_test_metrics
from utils.utils import new_log, to_cuda, to_cuda_inplace, detach_tensors_in_dict, seed_all, get_model_kwargs, model_dict

from utils.utils import get_fnames_labs_reg, get_fnames_unlab_reg
from utils.plot import plot_2dmatrix, plot_and_save
from utils.datasampler import LabeledUnlabeledSampler
from utils.constants import img_rows, img_cols, all_patches_mixed_train_part1, all_patches_mixed_test_part1, pop_map_root, inference_patch_size, overlap
from utils.constants import inference_patch_size as ips

import nvidia_smi
nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # check if we are doing domain adaptation or not
        if args.adversarial or args.CORAL or args.MMD:
            self.args.da = True
        else:
            self.args.da = False

        # set up dataloaders
        self.dataloaders = self.get_dataloaders(args)
        
        # set up model
        seed_all(args.seed)

        # define input channels based on the number of input modalities
        input_channels = args.Sentinel1*2  + args.NIR*1 + args.Sentinel2*3 + args.VIIRS*1

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

        # set up optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

        # set up info
        self.info = { "epoch": 0,  "iter": 0,  "sampleitr": 0}
        self.info["alpha"], self.info["beta"] = 0.0, 0.0
        self.train_stats, self.val_stats = defaultdict(lambda: np.nan), defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        # in case of checkpoint resume
        if args.resume is not None:
            self.resume(path=args.resume)


    def train(self):
        """
        Main training loop
        """
        with tqdm(range(self.info["epoch"], self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                # in domain validation
                if (self.info["epoch"] + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()
                    torch.cuda.empty_cache()
                
                # target domain testing
                if (self.info["epoch"] + 1) % (1*self.args.val_every_n_epochs) == 0:
                    self.test(plot=((self.info["epoch"]+1) % 20)==0, full_eval=((self.info["epoch"]+1) % 10)==0, zh_eval=True) #ZH
                    torch.cuda.empty_cache()
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
        self.train_stats = defaultdict(float)

        # set model to train mode
        self.model.train()

        # get GPU memory usage
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        self.train_stats["gpu_used"] = info.used / 1e9 # in GB 

        # check if we are in unsupervised or supervised mode and adjust dataloader accordingly
        dataloader = self.dataloaders['train']
        total = len(dataloader) if self.args.supmode=="unsup" else len(self.dataloaders["weak_target_dataset"])

        with tqdm(dataloader, leave=False, total=total) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)

            # iterate over samples of one epoch
            for i, sample in enumerate(inner_tnr):
                self.optimizer.zero_grad()
                optim_loss = 0.0
                loss_dict_weak = {}

                #  check if sample is weakly target supervised or source supervised 
                if self.args.supmode=="weaksup":
                    
                    # get weakly target supervised sample
                    try:
                        sample_weak = next(self.dataloaders["weak_target_iter"])
                    except StopIteration:
                        self.dataloaders["weak_target_iter"] = iter(self.dataloaders["weak_target"])
                        sample_weak = next(self.dataloaders["weak_target_iter"])

                    # forward pass and loss computation 
                    sample_weak = to_cuda_inplace(sample_weak)
                    output_weak = self.model(sample_weak, train=True, alpha=0., return_features=False, padding=False)

                    # merge augmented samples
                    if self.args.weak_merge_aug:
                        output_weak["popcount"] = output_weak["popcount"].sum(dim=0, keepdim=True)
                        sample_weak["y"] = sample_weak["y"].sum(dim=0, keepdim=True)
                        sample_weak["source"] = sample_weak["source"][0]

                    loss_weak, loss_dict_weak = get_loss(
                        output_weak, sample_weak, tag="weak", loss=args.loss, lam=args.lam, merge_aug=args.merge_aug)
                    
                    # Detach tensors
                    loss_dict_weak = detach_tensors_in_dict(loss_dict_weak)

                    # update loss
                    optim_loss += loss_weak * self.args.lam_weak * self.info["beta"]

                # forward pass
                sample = to_cuda_inplace(sample)
                output = self.model(sample, train=True, alpha=self.info["alpha"] if self.args.adversarial else 0., return_features=self.args.da)
            
                # compute loss
                loss, loss_dict = get_loss(output, sample, loss=args.loss, lam=args.lam, merge_aug=args.merge_aug,
                                           lam_adv=args.lam_adv if self.args.adversarial else 0.0,
                                           lam_coral=args.lam_coral if self.args.CORAL*self.info["beta"] else 0.0,
                                           lam_mmd=args.lam_mmd if self.args.MMD*self.info["beta"] else 0.0 )
                
                # update loss
                optim_loss += loss

                # Detach tensors
                loss_dict = detach_tensors_in_dict(loss_dict)

                # accumulate statistics 
                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key].cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                for key in loss_dict_weak:
                    self.train_stats[key] += loss_dict_weak[key].cpu().item() if torch.is_tensor(loss_dict_weak[key]) else loss_dict_weak[key]

                # detect NaN loss 
                if torch.isnan(loss):
                    raise Exception("detected NaN loss..")
                
                # backprop and stuff
                if self.info["epoch"] > 0 or not self.args.skip_first:
                    optim_loss.backward()

                    # gradient clipping
                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                    self.optimizer.step()
                    optim_loss = optim_loss.detach()
                    # torch.cuda.empty_cache()

                # update info
                self.info["iter"] += 1
                self.info["sampleitr"] += self.args.batch_size//2 if self.args.da else self.args.batch_size

                # update alpha for the adversarial loss, an annealing (to 1.0) schedule for alpha
                if self.args.adversarial:
                    self.info["alpha"] = self.update_param(float(i + self.info["epoch"] * len(self.dataloaders['train'])) / self.args.num_epochs / len(self.dataloaders['train']))

                # if we are in supervised mode, we need to update the weak data iterator when it runs out of data
                if self.args.supmode=="weaksup":
                    self.info["beta"] = self.update_param(float(i + self.info["epoch"] * len(self.dataloaders['train'])) / self.args.num_epochs / len(self.dataloaders['train']))
                    if self.dataloaders["weak_indices"][i%len(self.dataloaders["weak_indices"])] == self.dataloaders["weak_indices"][-1]:  
                        random.shuffle(self.dataloaders["weak_indices"]); self.log_train(); break

                # logging and stuff
                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.log_train((inner_tnr, tnr))
    
    def update_param(self, p, k=10):
        return 2. / (1. + np.exp(-k * p)) - 1

    def log_train(self, tqdmstuff=None):
        self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

        # print logs to console via tqdm
        if tqdmstuff is not None:
            inner_tnr, tnr = tqdmstuff
            inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
            if tnr is not None:
                tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                validation_loss=self.val_stats['optimization_loss'],
                                best_validation_loss=self.best_optimization_loss)

        # upload logs to wandb
        wandb.log({**{k + '/train': v for k, v in self.train_stats.items()}, **self.info}, self.info["iter"])
        
        # reset metrics
        self.train_stats = defaultdict(float)

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            pred, gt = [], []
            for sample in tqdm(self.dataloaders["val"], leave=False):

                # forward pass
                sample = to_cuda_inplace(sample)
                output = self.model(sample)
                
                # Colellect predictions and samples
                pred.append(output["popcount"].view(-1)); gt.append(sample["y"].view(-1))
                
                # compute loss
                loss, loss_dict = get_loss(output, sample, loss=args.loss, lam=args.lam, merge_aug=args.merge_aug, 
                                           lam_adv=args.lam_adv if self.args.adversarial else 0.0,
                                           lam_coral=args.lam_coral if self.args.CORAL else 0.0,
                                           lam_mmd=args.lam_mmd if self.args.MMD else 0.0,
                                           )

                # accumulate stats
                for key in loss_dict:
                    self.val_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 
            
            # Compute average metrics
            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            # Compute non-averagable metrics
            self.val_stats["Population/r2"] = r2(torch.cat(pred), torch.cat(gt))

            wandb.log({**{k + '/val': v for k, v in self.val_stats.items()}, **self.info}, self.info["iter"])
            
            # save best model
            if self.val_stats['optimization_loss'] < self.best_optimization_loss:
                self.best_optimization_loss = self.val_stats['optimization_loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')


    def test(self, plot=False, full_eval=False, zh_eval=True, save=False):
        self.test_stats = defaultdict(float)

        self.model.eval() 
        sum_pool10 = torch.nn.AvgPool2d(10, stride=10, divisor_override=1)
        sum_pool20 = torch.nn.AvgPool2d(20, stride=20, divisor_override=1)
        sum_pool40 = torch.nn.AvgPool2d(40, stride=40, divisor_override=1)
        sum_pool2 = torch.nn.AvgPool2d(2, stride=2, divisor_override=1)
        sum_pool4 = torch.nn.AvgPool2d(4, stride=4, divisor_override=1)

        if plot:
            print("Plotting predictions...")

        s = 0
        pad = torch.ones(1, 100,100)

        # Iterate though the test set and compute metrics
        with torch.no_grad():
            pred, gt = [], []
            pred1, gt1 = [], []
            pred2, gt2 = [], []
            pred4, gt4 = [], []
            pred10, gt10, gtSo2 = [], [], []
            for sample in tqdm(self.dataloaders["test"], leave=False):

                # forward pass
                sample = to_cuda_inplace(sample)
                output = self.model(sample)
                
                # Colellect predictions and samples
                if full_eval:
                    pred.append(output["popcount"].view(-1)); gt.append(sample["y"].view(-1)) 
                    loss, loss_dict = get_loss(output, sample, loss=args.loss, merge_aug=args.merge_aug, 
                                                lam_adv=args.lam_adv if self.args.adversarial else 0.0,
                                                lam_coral=args.lam_coral if self.args.CORAL else 0.0,
                                                lam_mmd=args.lam_mmd if self.args.MMD else 0.0,
                                               )

                    for key in loss_dict:
                        self.test_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                
                #fine_eval for Zurich
                if zh_eval:
                    if sample["pop_avail"].any(): 
                        pred_zh = output["popdensemap"][sample["pop_avail"][:,0].bool()]
                        gt_zh = sample["Pop_X"][sample["pop_avail"][:,0].bool()]
                        PopNN_X = sample["PopNN_X"][sample["pop_avail"][:,0].bool()]

                        # Collect all different aggregation scales  (1, 2, 4, 10)
                        pred1.append(sum_pool10(pred_zh).view(-1))
                        gt1.append(gt_zh.view(-1))
                        pred2.append(sum_pool20(pred_zh).view(-1))
                        gt2.append(sum_pool2(gt_zh).view(-1))
                        pred4.append(sum_pool40(pred_zh).view(-1))
                        gt4.append(sum_pool4(gt_zh).view(-1))

                        gt10.append(sum_pool10(gt_zh).view(-1))
                        pred10.append(output["popcount"][sample["pop_avail"][:,0].bool()].view(-1))
                        gtSo2.append(sample["y"][sample["pop_avail"][:,0].bool()].view(-1))

                        # Plot predictions for Zurich
                        i = 0
                        if plot:
                            for i in range(len(gt_zh)):
                                if s>230:
                                    vmax = max([gt_zh[i].max(), pred_zh[i].max()*100]).cpu().item()
                                    plot_and_save(gt_zh[i].cpu(), model_name=args.expN, title=gt_zh[i].sum().cpu().item(), vmin=0, vmax=vmax, idx=s, name="01_GT", folder=self.args.experiment_folder)
                                    plot_and_save(sum_pool10(pred_zh)[i].cpu(), model_name=args.expN, title=sum_pool10(pred_zh)[i].sum().cpu().item(), vmin=0, vmax=vmax, idx=s, name="02_pred10", folder=self.args.experiment_folder)
                                    # plot_and_save(PopNN_X[i].cpu(), model_name=args.expN, title=(PopNN_X[i].sum()/100).cpu().item(), vmin=0, vmax=vmax, idx=s, name="03_GTNN", folder=self.args.experiment_folder)
                                    plot_and_save(pred_zh[i].cpu()*100, model_name=args.expN, title=pred_zh[i].sum().cpu().item(), vmin=0, vmax=vmax, idx=s, name="04_pred100", folder=self.args.experiment_folder)

                                    inp = sample["input"][sample["pop_avail"][:,0].bool()]
                                    if args.Sentinel2:
                                        plot_and_save(inp[i,:3].cpu().permute(1,2,0)*0.2+0.5, model_name=args.expN, title=self.args.expN, idx=s, name="05_S2", cmap=None, folder=self.args.experiment_folder)
                                        if args.Sentinel1:
                                            plot_and_save(torch.cat([inp[i,3:5].cpu()*0.4 + 0.3, pad]).permute(1,2,0), model_name=args.expN, title=self.args.expN, idx=s, name="06_S1", cmap=None, folder=self.args.experiment_folder)
                                    else:
                                        plot_and_save(torch.cat([inp[i,:2].cpu()*0.4 + 0.3, pad]).permute(1,2,0), model_name=args.expN, title=self.args.expN, idx=s, name="06_S1", cmap=None, folder=self.args.experiment_folder)

                                s += 1
                                if s > 270:
                                    break
            
            # Compute metrics for full test set
            if full_eval:
                # average all stats
                self.test_stats = {k: v / len(self.dataloaders['val']) for k, v in self.test_stats.items()}

                # Compute non-averagable metrics
                self.test_stats["Population/r2"] = r2(torch.cat(pred), torch.cat(gt))
                self.test_stats["Population/Correlation"] = r2(torch.cat(pred), torch.cat(gt))
                wandb.log({**{k + '/test': v for k, v in self.test_stats.items()}, **self.info}, self.info["iter"])

            #fine_eval for Zurich
            if zh_eval:
                self.test_stats1 = get_test_metrics(torch.cat(pred1), torch.cat(gt1), tag="100m")
                self.test_stats2 = get_test_metrics(torch.cat(pred2), torch.cat(gt2), tag="200m")
                self.test_stats4 = get_test_metrics(torch.cat(pred4), torch.cat(gt4), tag="400m")
                self.test_stats10 = get_test_metrics(torch.cat(pred10), torch.cat(gt10), tag="1km")
                self.test_statsGT = get_test_metrics(torch.cat(gt10), torch.cat(gtSo2), tag="GTCons")
                self.test_statsZH = {**self.test_stats1, **self.test_stats2, **self.test_stats4, **self.test_stats10, **self.test_statsGT}
            
                wandb.log({**{k + '/testZH': v for k, v in self.test_statsZH.items()}, **self.info}, self.info["iter"])

    def test_target(self, save=False):
        # Test on target domain
        self.model.eval()
        self.test_stats = defaultdict(float)

        with torch.no_grad(): 
            for testdataloader in self.dataloaders["test_target"]:

                # inputialize the output map
                h, w = testdataloader.dataset.shape()
                output_map = torch.zeros((h, w))
                output_map_count = torch.zeros((h, w))

                for sample in tqdm(testdataloader, leave=False):
                    sample = to_cuda_inplace(sample)

                    # get the valid coordinates
                    xmin, xmax, ymin, ymax = [val.item() for val in sample["valid_coords"]]
                    xl,yl = [val.item() for val in sample["img_coords"]]
                    mask = sample["mask"][0].bool()

                    # get the output with a forward pass
                    output = self.model(sample, padding=False)

                    # add the output to the output map
                    output_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["popdensemap"][0][mask].cpu()
                    output_map_count[xl:xl+ips, yl:yl+ips][mask.cpu()] += 1

                # average over the number of times each pixel was visited
                output_map[output_map_count>0] = output_map[output_map_count>0] / output_map_count[output_map_count>0]

                if save:
                    # save the output map
                    testdataloader.dataset.save(output_map, self.experiment_folder)
                
                # convert populationmap to census
                census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map, gpu_mode=True)
                self.target_test_stats = get_test_metrics(census_pred, census_gt.float().cuda(), tag="census")
                
            wandb.log({**{k + '/targettest': v for k, v in self.target_test_stats.items()}, **self.info}, self.info["iter"])
        

    @staticmethod
    def get_dataloaders(args, force_recompute=False): 
        """
        Get dataloaders for the source and target domains
        Inputs:
            args: command line arguments
            force_recompute: if True, recompute the dataloaders even if they already exist
        Outputs:
            dataloaders: dictionary of dataloaders
                """

        input_defs = {'S1': args.Sentinel1, 'S2': args.Sentinel2, 'VIIRS': args.VIIRS, 'NIR': args.NIR}
        params = {'dim': (img_rows, img_cols), "satmode": args.satmode, 'in_memory': args.in_memory, **input_defs}

        if args.full_aug:
                data_transform = transforms.Compose([
                    AddGaussianNoise(std=0.1, p=0.9),
                    # SyntheticHaze(p=0.75),
                    # RandomHorizontalVerticalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5),
                    RandomRotationTransform(angles=[90, 180, 270], p=0.75),
                ])
        else: 
            if not args.Sentinel1: 
                data_transform = transforms.Compose([
                    HazeAdditionModule()
                    # AddGaussianNoise(std=0.1, p=0.9),
                    # transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
                    # RandomRotationTransform(angles=[90, 180, 270], p=0.75), 
                ])
            else:
                data_transform = transforms.Compose([
                    # AddGaussianNoise(std=0.1, p=0.9),
                    # RandomHorizontalVerticalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
                    # RandomRotationTransform(angles=[90, 180, 270], p=0.75), 
                ])
        
        # source domain samples
        val_size = 0.2 
        f_names, labels = get_fnames_labs_reg(all_patches_mixed_train_part1, force_recompute=force_recompute)

        # remove elements that contain "zurich" as a substring
        if args.excludeZH:
            f_namesX = []
            labelsX = []
            [(f_namesX.append(f),labelsX.append(l)) for f,l in zip(f_names,labels) if "zurich" not in f]
            f_names, labels = f_namesX, labelsX

        # limit the number of samples for debugging
        f_names, labels = f_names[:int(args.max_samples)] , labels[:int(args.max_samples)]
        s = int(len(f_names)*val_size)
        f_names_train, f_names_val, labels_train, labels_val = f_names[:-s], f_names[-s:], labels[:-s], labels[-s:]
        f_names_test, labels_test = get_fnames_labs_reg(all_patches_mixed_test_part1, force_recompute=False)

        # unlabled target domain samples
        if args.da:
            f_names_unlab = []
            for reg in args.target_regions:
                f_names_unlab.extend(get_fnames_unlab_reg(os.path.join(pop_map_root, os.path.join("EE", reg)), force_recompute=force_recompute))
        else:
            f_names_unlab = []

        # create the raw source dataset
        train_dataset = PopulationDataset_Reg(f_names_train, labels_train, f_names_unlab=f_names_unlab, mode="train",
                                            transform=data_transform,random_season=args.random_season, **params)
        datasets = {
            "train": train_dataset,
            "val": PopulationDataset_Reg(f_names_val, labels_val, mode="val", transform=None, **params),
            "test": PopulationDataset_Reg(f_names_test, labels_test, mode="test", transform=None, **params),
            "test_target": [ Population_Dataset_target(reg, patchsize=ips, overlap=overlap, **input_defs) for reg in args.target_regions ]
        }
        
        # create the datasampler for the source/target domain mixup
        custom_sampler, shuffle = None, True 
        if len(args.target_regions)>0 and len(datasets["train"].unlabeled_indices)>0:
            custom_sampler = LabeledUnlabeledSampler( labeled_indices=datasets["train"].labeled_indices, unlabeled_indices=datasets["train"].unlabeled_indices,
                                                       batch_size=args.batch_size  )
            shuffle = False

        # create the dataloaders
        dataloaders =  {
            "train": DataLoader(datasets["train"], batch_size=args.batch_size, num_workers=args.num_workers, sampler=custom_sampler, shuffle=shuffle, drop_last=True, pin_memory=True),
            "val":  DataLoader(datasets["val"], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, pin_memory=True),
            "test":  DataLoader(datasets["test"], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False,pin_memory=True),
            "test_target":  [DataLoader(datasets["test_target"], batch_size=1, num_workers=1, shuffle=False, drop_last=False) for datasets["test_target"] in datasets["test_target"] ]
        }
        

        # add weakly supervised samples of the target domain to the trainind_dataset
        if args.supmode=="weaksup":
            # create the weakly supervised dataset stack them into a single dataset and dataloader
            weak_batchsize = 2 if args.weak_merge_aug else 1
            weak_datasets = []
            for reg in args.target_regions:
                weak_datasets.append( Population_Dataset_target(reg, mode="weaksup", patchsize=None, overlap=None, max_samples=args.max_weak_samples,
                                                                fourseasons=args.random_season, transform=data_transform, **input_defs)  )
            dataloaders["weak_target_dataset"] = ConcatDataset(weak_datasets)
            
            # create own simulation of a dataloader for the weakdataset
            weak_indices = list(range(len(dataloaders["weak_target_dataset"])))
            random.shuffle(weak_indices)
            dataloaders["weak_indices"] = weak_indices
            dataloaders["weak_iter"] = itertools.cycle(weak_indices)

            # create dataloader for the weakly supervised dataset
            dataloaders["weak_target"] = DataLoader(dataloaders["weak_target_dataset"], batch_size=weak_batchsize, num_workers=1, shuffle=True, collate_fn=Population_Dataset_collate_fn, drop_last=True)
            dataloaders["weak_target_iter"] = iter(dataloaders["weak_target"])
        
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
