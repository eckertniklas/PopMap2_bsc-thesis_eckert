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
from utils.transform import Eu2Rwa
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
from data.So2Sat import PopulationDataset_Reg
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
from utils.utils import get_fnames_labs_reg, get_fnames_unlab_reg
from utils.datasampler import LabeledUnlabeledSampler
from utils.constants import img_rows, img_cols, all_patches_mixed_train_part1, all_patches_mixed_test_part1, pop_map_root, testlevels, overlap
from utils.constants import inference_patch_size as ips
from utils.utils import Namespace

from model.cycleGAN.models import create_model
from model.cycleGAN.util.visualizer import Visualizer
from model.cycleGAN.util.util import tensor2im

torch.autograd.set_detect_anomaly(True)

import nvidia_smi
nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # check if we are doing domain adaptation or not
        if args.adversarial or args.CORAL or args.MMD or self.args.CyCADA:
            self.args.da = True
        else:
            self.args.da = False

        if args.loss in ["gaussian_nll", "log_gaussian_nll", "laplacian_nll", "log_laplacian_nll", "gaussian_aug_loss", "log_gaussian_aug_loss", "laplacian_aug_loss", "log_laplacian_aug_loss"]:
            self.args.probabilistic = True
        else:
            self.args.probabilistic = False

        # set up dataloaders
        self.dataloaders = self.get_dataloaders(self, args)
        
        # set up model
        seed_all(args.seed)

        # define input channels based on the number of input modalities
        # input_channels = args.Sentinel1*2  + args.NIR*1 + args.Sentinel2*3 + args.VIIRS*1
        
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

        if args.CyCADA:
            # load the pretrained cycleGAN model 
            self.opt = Namespace(model="cycle_gan", name=args.CyCADAGANcheckpoint, input_nc=5, output_nc=5, ngf=64, ndf=64,
                            netG=args.CyCADAnetG, netD='basic', n_layers_D=3, norm='instance', windows_size=100, direction='AtoB',
                            no_dropout=True, init_type="normal", init_gain=0.02, epoch="latest", load_iter=0, isTrain=True, gpu_ids=[0],
                            preprocess=None, continue_train=self.args.CyCADAcontinue, gan_mode="lsgan", pool_size=50, beta1=0.5, lambda_identity=0.5, lr=0.0002, 
                            dataset_mode="unaligned", verbose=False, lr_policy="linear", epoch_count=1, n_epochs=args.num_epochs, n_epochs_decay=args.num_epochs,
                            lambda_A=10.0, lambda_B=10.0, lambda_consistency_fake_B=args.lambda_consistency_fake_B, lambda_consistency_real_B=args.lambda_consistency_real_B,
                            lambda_popB=args.lambda_popB, display_freq=400, save_latest_freq=2500, save_by_iter=False,
                            display_id=0, no_html=True, display_port=8097, update_html_freq=1000,
                            use_wandb=True, display_ncols=4,  wandb_project_name="CycleGAN-and-pix2pix", display_winsize=100, display_env="main", display_server="http://localhost",
                            # model_suffix="_A",
                            checkpoints_dir= args.save_dir)
                            # checkpoints_dir= os.path.join(args.save_dir, "checkpointsCyCADA/") )
                            # checkpoints_dir="/scratch2/metzgern/HAC/code/CycleGANAugs/pytorch-CycleGAN-and-pix2pix/checkpoints/" )
            self.CyCADAmodel = create_model(self.opt)      # create a model given opt.model and other options
            self.CyCADAmodel.setup(self.opt)               # regular setup: load and print networks; create schedulers 

            # load the pretrained population model
            if args.model in model_dict:
                # get the source model and load weights
                model_kwargs = get_model_kwargs(args, args.model)
                self.CyCADAmodel.sourcepopmodel = model_dict[args.model](**model_kwargs).cuda()
                self.CyCADAmodel.sourcepopmodel.load_state_dict(torch.load(args.CyCADASourcecheckpoint)['model'])
                self.CyCADAmodel.sourcepopmodel.eval()

                self.visualizer = Visualizer(self.opt)   # create a visualizer that display/save images and plots
                total_iters = 0                # the total number of training iterations

                # initialize the target model
                self.model.load_state_dict(torch.load(args.CyCADASourcecheckpoint)['model']) 
                
            else:
                raise ValueError(f"Unknown model: {args.model}")

        if args.adversarial:
            self.model.domain_classifier = self.model.get_domain_classifier(args.feature_dim, args.classifier).cuda()
            
        # number of params
        args.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Model", args.model, "; #Params:", args.pytorch_total_params)

        # set up experiment folder
        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, "So2Sat"), args)
        self.args.experiment_folder = self.experiment_folder
        
        # wandb config
        wandb.init(project=args.wandb_project, dir=self.experiment_folder)
        wandb.config.update(self.args) 
        
        # set up model
        seed_all(args.seed+2)

        # set up optimizer and scheduler
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
        elif args.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
            # self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weightdecay)

        if args.stochasticWA:
            self.optimizer = SWA(self.optimizer, swa_start=2, swa_freq=1, swa_lr=0.05)
            self.optimizer.optimizer.defaults = None


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

    def train(self):
        """
        Main training loop
        """
        with tqdm(range(self.info["epoch"], self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                
                self.train_epoch(tnr)
                torch.cuda.empty_cache()

                # in domain validation
                if (self.info["epoch"] + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()
                    torch.cuda.empty_cache()

                    # TODO weak validation
                    # if self.args.supmode=="weaksup":
                    #     self.validate_weak()
                    #     torch.cuda.empty_cache()
                
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
        train_stats = defaultdict(float)

        # set model to train mode
        self.model.train()

        # get GPU memory usage
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        train_stats["gpu_used"] = info.used / 1e9 # in GB 

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
                loss_dict_raw = {}
                loss_CyCADAtarget = {}

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
                    sample_weak = apply_transformations_and_normalize(sample_weak, self.data_transform, self.dataset_stats, buildinginput=self.args.buildinginput)
                    # print(sample_weak["input"].shape[2:])
                    # check if the input is to large
                    if sample_weak["input"].shape[2]*sample_weak["input"].shape[3] > 1400000:
                        encoder_no_grad, unet_no_grad = True, False
                        if sample_weak["input"].shape[2]*sample_weak["input"].shape[3] > 6000000:
                            encoder_no_grad, unet_no_grad = True, True 
                            if sample_weak["input"].shape[2]*sample_weak["input"].shape[3] > 12000000:
                                print("Input to large for encoder and unet")
                                continue 
                    else:
                        encoder_no_grad, unet_no_grad = False, False 

                    output_weak = self.model(sample_weak, train=True, alpha=0., return_features=False, padding=False, encoder_no_grad=encoder_no_grad, unet_no_grad=unet_no_grad)

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

                    loss_weak, loss_dict_weak = get_loss(
                        output_weak, sample_weak, tag="weak", loss=args.loss, lam=args.lam, merge_aug=args.merge_aug)
                    
                    # Detach tensors
                    loss_dict_weak = detach_tensors_in_dict(loss_dict_weak)
                    
                    if self.boosted:
                        boosted_loss = [el.replace("gaussian", "l1") if el in ["gaussian_nll", "log_gaussian_nll", "gaussian_aug_loss", "log_gaussian_aug_loss"] else el for el in args.loss]
                        boosted_loss = [el.replace("laplace", "l1") if el in ["laplacian_nll", "log_laplacian_nll", "laplace_aug_loss", "log_laplace_aug_loss"] else el for el in boosted_loss]
                        loss_weak_raw, loss_weak_dict_raw = get_loss(
                            output_weak["intermediate"], sample_weak, loss=boosted_loss, lam=args.lam, merge_aug=args.merge_aug, tag="train_weak_intermediate")
                        
                        loss_weak_dict_raw = detach_tensors_in_dict(loss_weak_dict_raw)
                        loss_dict_weak = {**loss_dict_weak, **loss_weak_dict_raw}
                        loss_weak += loss_weak_raw

                    # update loss
                    optim_loss += loss_weak * self.args.lam_weak #* self.info["beta"]
                else:
                    output_weak = None

                # forward pass
                sample = to_cuda_inplace(sample)

                if self.args.CyCADA:
                    # CyCADA forward pass and loss computation
                    sample = apply_transformations_and_normalize(sample, self.data_transform, self.dataset_stats, buildinginput=self.args.buildinginput)
                    data = {"A": sample["input"][sample["source"]], "B": sample["input"][~sample["source"]], "A_paths": "", "B_paths": ""}
                    self.CyCADAmodel.set_input(data)         # unpack data from dataset and apply preprocessing
                    self.CyCADAmodel.optimize_parameters(gt=sample["y"][sample["source"]])

                    # replace all existing source domain samples with fake target domain samples
                    sample["input"] = torch.cat([self.CyCADAmodel.fake_B_prep.detach(), self.CyCADAmodel.real_B_prep.detach()], dim=0)

                else:
                    if not self.args.GANonly and not self.args.nomain: 
                        sample = apply_transformations_and_normalize(sample, self.data_transform, self.dataset_stats, buildinginput=self.args.buildinginput)
                

                if not self.args.GANonly and not self.args.nomain:
                    # forward pass for the main model
                    output = self.model(sample, train=True, alpha=self.info["alpha"] if self.args.adversarial else 0., return_features=self.args.da)
                
                    # compute loss
                    loss, loss_dict = get_loss(output, sample, loss=args.loss, lam=args.lam, merge_aug=args.merge_aug,
                                            lam_adv=args.lam_adv if self.args.adversarial else 0.0,
                                            lam_coral=args.lam_coral if self.args.CORAL else 0.0,
                                            lam_mmd=args.lam_mmd if self.args.MMD else 0.0,
                                            tag="train_main")
                    if self.boosted:
                        boosted_loss = [el.replace("gaussian", "l1") if el in ["gaussian_nll", "log_gaussian_nll", "gaussian_aug_loss", "log_gaussian_aug_loss"] else el for el in args.loss]
                        boosted_loss = [el.replace("laplace", "l1") if el in ["laplacian_nll", "log_laplacian_nll", "laplace_aug_loss", "log_laplace_aug_loss"] else el for el in boosted_loss]
                        loss_raw, loss_dict_raw = get_loss(output["intermediate"], sample, loss=boosted_loss, lam=args.lam, merge_aug=args.merge_aug,
                                            lam_adv=args.lam_adv if self.args.adversarial else 0.0,
                                            lam_coral=args.lam_coral if self.args.CORAL else 0.0,
                                            lam_mmd=args.lam_mmd if self.args.MMD else 0.0,
                                            tag="train_intermediate")
                        
                        loss += loss_raw * self.args.lam_raw
                        loss_dict_raw = detach_tensors_in_dict(loss_dict_raw)
                        # loss_dict = {**loss_dict, **loss_dict_raw}
                
                    # update loss
                    optim_loss += loss
                
                    # consistency losses for the target model with the outputs of the cycleGAN
                    if self.args.CyCADA:

                        # pixel supervision to the source domain model
                        loss_CyCADAtarget["loss_targetconsistency"] = self.CyCADAmodel.criterionFakePop(output["popdensemap"][sample["source"]], self.CyCADAmodel.real_A_output["popdensemap"].detach())
                        loss_CyCADAtarget["loss_targetconsistency_lam"] = loss_CyCADAtarget["loss_targetconsistency"] * self.args.lam_targetconsistency
                        optim_loss += loss_CyCADAtarget["loss_targetconsistency_lam"]

                        # selfsupervised loss
                        loss_CyCADAtarget["loss_selfsupervised_consistency"] = self.CyCADAmodel.criterionFakePop(output["popdensemap"][~sample["source"]], self.CyCADAmodel.fake_A_output["popdensemap"].detach())
                        loss_CyCADAtarget["loss_selfsupervised_consistency_lam"] = loss_CyCADAtarget["loss_selfsupervised_consistency"] * self.args.lam_selfsupervised_consistency
                        optim_loss += loss_CyCADAtarget["loss_selfsupervised_consistency_lam"]
                else:
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

                # detect NaN loss 
                # backprop and stuff
                if not self.args.GANonly:
                    if torch.isnan(optim_loss):
                        raise Exception("detected NaN loss..")
                    
                    if self.info["epoch"] > 0 or not self.args.no_opt:
                        # backprop
                        optim_loss.backward()

                        # gradient clipping
                        if self.args.gradient_clip > 0.:
                            clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                        self.optimizer.step()
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
                self.info["sampleitr"] += self.args.batch_size//2 if self.args.da else self.args.batch_size

                # update alpha for the adversarial loss, an annealing (to 1.0) schedule for alpha
                if self.args.adversarial:
                    self.info["alpha"] = self.update_param(float(i + self.info["epoch"] * len(self.dataloaders['train'])) / self.args.num_epochs / len(self.dataloaders['train']))

                # if we are in supervised mode, we need to update the weak data iterator when it runs out of data
                if self.args.supmode=="weaksup":
                    self.info["beta"] = self.update_param(float(i + self.info["epoch"] * len(self.dataloaders['train'])) / self.args.num_epochs / len(self.dataloaders['train']))
                    if self.dataloaders["weak_indices"][i%len(self.dataloaders["weak_indices"])] == self.dataloaders["weak_indices"][-1]:  
                        random.shuffle(self.dataloaders["weak_indices"]); self.log_train(train_stats); break

                if self.args.CyCADA:
                    if self.info["sampleitr"] % self.opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                        save_result = self.info["sampleitr"] % self.opt.update_html_freq == 0
                        self.CyCADAmodel.compute_visuals()
                        self.visualizer.display_current_results(self.CyCADAmodel.get_current_visuals(), self.info["epoch"], save_result)

                    # if self.info["sampleitr"] % self.opt.print_freq == 0:    # print training losses and save logging information to the disk
                    #     losses = self.CyCADAmodel.get_current_losses()

                    if self.info["sampleitr"] % self.opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                        print('saving the latest model (epoch %d, total_iters %d)' % (self.info["epoch"], self.info["sampleitr"]))
                        save_suffix = 'iter_%d' % self.info["sampleitr"] if self.opt.save_by_iter else 'latest'
                        self.CyCADAmodel.save_networks(save_suffix)

                # logging and stuff
                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    
                    if self.args.CyCADA:
                        losses_CyCADA = self.CyCADAmodel.get_current_losses()
                        losses_CyCADA = detach_tensors_in_dict({**losses_CyCADA,**loss_CyCADAtarget})
                        for key in losses_CyCADA:
                            train_stats[key] += losses_CyCADA[key].cpu().item() if torch.is_tensor(losses_CyCADA[key]) else losses_CyCADA[key]
                        train_stats["optimization_loss"] = self.CyCADAmodel.loss_cycle_A
                    
                    log_target_img = True
                    if log_target_img and not self.args.GANonly:
                        if self.args.CyCADA:
                            wandb_image = wandb.Image(tensor2im(output["popdensemap"][sample["source"]].unsqueeze(1)-0.5))
                            wandb.log({"fake_B_target_popdensemap": wandb_image}, step=self.info["iter"])
                            wandb_image = wandb.Image(tensor2im(output["popdensemap"][~sample["source"]].unsqueeze(1)-0.5))
                            wandb.log({"real_B_target_popdensemap": wandb_image}, step=self.info["iter"])
                    train_stats = self.log_train(train_stats,(inner_tnr, tnr))
                    train_stats = defaultdict(float)
        
        if self.args.stochasticWA:
            self.optimizer.swap_swa_sgd()
    
    def update_param(self, p, k=10):
        return 2. / (1. + np.exp(-k * p)) - 1

    def log_train(self, train_stats, tqdmstuff=None):
        train_stats = {k: v / self.args.logstep_train for k, v in train_stats.items()}

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
        

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            pred, gt = [], []
            for sample in tqdm(self.dataloaders["val"], leave=False):

                # forward pass
                sample = to_cuda_inplace(sample)
                sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput)

                output = self.model(sample)
                
                # Colellect predictions and samples
                pred.append(output["popcount"].view(-1)); gt.append(sample["y"].view(-1))
                
                # compute loss
                loss, loss_dict = get_loss(output, sample, loss=args.loss, lam=args.lam, merge_aug=args.merge_aug, 
                                           lam_adv=args.lam_adv if self.args.adversarial else 0.0,
                                           lam_coral=args.lam_coral if self.args.CORAL else 0.0,
                                           lam_mmd=args.lam_mmd if self.args.MMD else 0.0,
                                           tag="val_main")
                if self.boosted:
                    loss_raw, loss_dict_raw = get_loss(output["intermediate"], sample, loss=args.loss, lam=args.lam, merge_aug=args.merge_aug,
                                           lam_adv=args.lam_adv if self.args.adversarial else 0.0,
                                           lam_coral=args.lam_coral if self.args.CORAL else 0.0,
                                           lam_mmd=args.lam_mmd if self.args.MMD else 0.0,
                                           tag="val_intermediate")
                    loss_dict = {**loss_dict, **loss_dict_raw}

                # accumulate stats
                for key in loss_dict:
                    self.val_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 
            
                    
            # Compute average metrics
            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            # Compute non-averagable metrics
            self.val_stats["Population_val_main/r2"] = r2(torch.cat(pred), torch.cat(gt))

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
                sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput)
                # sample = apply_normalize(sample, self.dataset_stats)

                output = self.model(sample)
                
                # Colellect predictions and samples
                if full_eval:
                    pred.append(output["popcount"].view(-1)); gt.append(sample["y"].view(-1)) 
                    _, loss_dict = get_loss(output, sample, loss=args.loss, merge_aug=args.merge_aug, 
                                                lam_adv=args.lam_adv if self.args.adversarial else 0.0,
                                                lam_coral=args.lam_coral if self.args.CORAL else 0.0,
                                                lam_mmd=args.lam_mmd if self.args.MMD else 0.0,
                                               tag="test_main")
                    if self.boosted:
                        _, loss_dict_raw = get_loss(output["intermediate"], sample, loss=args.loss, lam=args.lam, merge_aug=args.merge_aug,
                                            lam_adv=args.lam_adv if self.args.adversarial else 0.0,
                                            lam_coral=args.lam_coral if self.args.CORAL else 0.0,
                                            lam_mmd=args.lam_mmd if self.args.MMD else 0.0,
                                            tag="test_intermediate")
                        loss_dict = {**loss_dict, **loss_dict_raw}

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
                self.test_stats["Population_test_main/r2"] = r2(torch.cat(pred), torch.cat(gt))
                self.test_stats["Population_test_main/Correlation"] =  torch.corrcoef(torch.stack([torch.cat(pred),torch.cat(gt)]))[0,1]
                # self.test_stats["Population/Correlation"] = corr(torch.cat(pred), torch.cat(gt))
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

    def test_target(self, save=False, full=True):
        # Test on target domain
        self.model.eval()
        self.test_stats = defaultdict(float)
        # self.model.train()

        with torch.no_grad(): 
            self.target_test_stats = defaultdict(float)
            for testdataloader in self.dataloaders["test_target"]:

                # inputialize the output map
                h, w = testdataloader.dataset.shape()
                output_map = torch.zeros((h, w), dtype=torch.float16)
                # census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map, gpu_mode=True, level=level)
                output_map_count = torch.zeros((h, w), dtype=torch.int8)

                if self.args.probabilistic:
                    output_map_var = torch.zeros((h, w), dtype=torch.float16)
                if self.boosted and full:
                    output_map_raw = torch.zeros((h, w), dtype=torch.float16)
                    if self.args.probabilistic:
                        output_map_var_raw = torch.zeros((h, w), dtype=torch.float16)

                for sample in tqdm(testdataloader, leave=False):
                    sample = to_cuda_inplace(sample)
                    # sample = apply_normalize(sample, self.dataset_stats)
                    sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput)

                    # get the valid coordinates
                    # xmin, xmax, ymin, ymax = [val.item() for val in sample["valid_coords"]]
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
                    # output_map_count[xl:xl+ips, yl:yl+ips][mask.cpu()] += 1

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

                if save:
                    # save the output map
                    testdataloader.dataset.save(output_map, self.experiment_folder)
                    if self.args.probabilistic:
                        testdataloader.dataset.save(output_map_var, self.experiment_folder, tag="VAR_{}".format(testdataloader.dataset.region))
                    if self.boosted and full:
                        testdataloader.dataset.save(output_map_raw, self.experiment_folder, tag="RAW_{}".format(testdataloader.dataset.region))
                        if self.args.probabilistic:
                            testdataloader.dataset.save(output_map_var_raw, self.experiment_folder, tag="VAR_RAW_{}".format(testdataloader.dataset.region))
                
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
    
                    scatterplot = scatter_plot3(census_pred.tolist(), census_gt.tolist(), log_scale=True)
                    if scatterplot is not None:
                        self.target_test_stats["Scatter/Scatter_{}_{}".format(testdataloader.dataset.region, level)] = wandb.Image(scatterplot)
                        # scatterplot.save("last_scatter.png")

            wandb.log({**{k + '/targettest': v for k, v in self.target_test_stats.items()}, **self.info}, self.info["iter"])
        

    @staticmethod
    def get_dataloaders(self, args, force_recompute=False): 
        """
        Get dataloaders for the source and target domains
        Inputs:
            args: command line arguments
            force_recompute: if True, recompute the dataloader's and look out for new files even if the file list already exist
        Outputs:
            dataloaders: dictionary of dataloaders
        """

        input_defs = {'S1': args.Sentinel1, 'S2': args.Sentinel2, 'VIIRS': args.VIIRS, 'NIR': args.NIR}
        params = {'dim': (img_rows, img_cols), "satmode": args.satmode, 'in_memory': args.in_memory, **input_defs}
        self.data_transform = {}
        if args.full_aug:
            self.data_transform["general"] = transforms.Compose([
                # AddGaussianNoise(std=0.04, p=0.75), 
                # RandomHorizontalVerticalFlip(p=0.5),
                RandomVerticalFlip(p=0.5, allsame=args.supmode=="weaksup"),
                RandomHorizontalFlip(p=0.5, allsame=args.supmode=="weaksup"),
                RandomRotationTransform(angles=[90, 180, 270], p=0.75),
            ])
            S2augs = [  RandomBrightness(p=0.9, beta_limit=(0.666, 1.5)),
                    RandomGamma(p=0.9, gamma_limit=(0.6666, 1.5)),
                    # HazeAdditionModule(p=0.5, atm_limit=(0.3, 1.0), haze_limit=(0.05,0.3))
            ]
            if args.eu2rwa:
                S2augs.append(Eu2Rwa(p=1.0))


        else: 
            self.data_transform["general"] = transforms.Compose([
                # AddGaussianNoise(std=0.1, p=0.9),
            ])
            S2augs = [  ]
        # ])
        # S2augs = [  RandomBrightness(p=0.9, beta_limit=(0.666, 1.5)),
        #             RandomGamma(p=0.9, gamma_limit=(0.6666, 1.5)),
        #             # HazeAdditionModule(p=0.5, atm_limit=(0.3, 1.0), haze_limit=(0.05,0.3))
        # ]
        # if args.eu2rwa:
        #     S2augs.append(Eu2Rwa(p=1.0))
        self.data_transform["S2"] = OwnCompose(S2augs)

        self.data_transform["S1"] = transforms.Compose([
            # RandomBrightness(p=0.95),
            # RandomGamma(p=0.95),
            # HazeAdditionModule(p=0.95)
        ])
        
        # load normalization stats
        self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats_unified_2A.json'))
        for mkey in self.dataset_stats.keys():
            if isinstance(self.dataset_stats[mkey], dict):
                for key,val in self.dataset_stats[mkey].items():
                    self.dataset_stats[mkey][key] = torch.tensor(val)
            else:
                self.dataset_stats[mkey] = torch.tensor(val)

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
        f_names_test, labels_test = get_fnames_labs_reg(all_patches_mixed_test_part1, force_recompute=force_recompute)

        # unlabled target domain samples
        if args.da:
            f_names_unlab = []
            for reg in args.target_regions:
                f_names_unlab.extend(get_fnames_unlab_reg(os.path.join(pop_map_root, os.path.join("EE", reg)), force_recompute=False))
        else:
            f_names_unlab = []

        # create the raw source dataset
        train_dataset = PopulationDataset_Reg(f_names_train, labels_train, f_names_unlab=f_names_unlab, mode="train",
                                            transform=None,random_season=args.random_season, **params)
        datasets = {
            "train": train_dataset,
            "val": PopulationDataset_Reg(f_names_val, labels_val, mode="val", transform=None, **params),
            "test": PopulationDataset_Reg(f_names_test, labels_test, mode="test", transform=None, **params),
            "test_target": [ Population_Dataset_target(reg, patchsize=ips, overlap=overlap, sentinelbuildings=args.sentinelbuildings, **input_defs) for reg in args.target_regions ]
        }
        
        # create the datasampler for the source/target domain mixup
        custom_sampler, shuffle = None, True 
        if len(args.target_regions)>0 and len(datasets["train"].unlabeled_indices)>0:
            custom_sampler = LabeledUnlabeledSampler( labeled_indices=datasets["train"].labeled_indices, unlabeled_indices=datasets["train"].unlabeled_indices,
                                                       batch_size=args.batch_size  )
            shuffle = False

        # create the dataloaders
        dataloaders =  {
            "train": DataLoader(datasets["train"], batch_size=args.batch_size, num_workers=args.num_workers, sampler=custom_sampler, shuffle=shuffle, drop_last=True, pin_memory=False),
            "val":  DataLoader(datasets["val"], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, pin_memory=True),
            "test":  DataLoader(datasets["test"], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, pin_memory=True),
            "test_target":  [DataLoader(datasets["test_target"], batch_size=1, num_workers=1, shuffle=False, drop_last=False) for datasets["test_target"] in datasets["test_target"] ]
        }
        
        # add weakly supervised samples of the target domain to the trainind_dataset
        if args.supmode=="weaksup":
            # create the weakly supervised dataset stack them into a single dataset and dataloader
            weak_batchsize = args.weak_batch_size
            # weak_batchsize = args.weak_batch_size if args.weak_merge_aug else 1
            weak_datasets = []
            for reg in args.target_regions_train:
                weak_datasets.append( Population_Dataset_target(reg, mode="weaksup", patchsize=None, overlap=None, max_samples=args.max_weak_samples,
                                                                fourseasons=args.random_season, transform=None, sentinelbuildings=args.sentinelbuildings, 
                                                                ascfill=True, **input_defs)  )
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
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])
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
