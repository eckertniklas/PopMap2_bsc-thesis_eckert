import os
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
from torch import is_tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision import transforms
from utils.transform import RandomRotationTransform, RandomHorizontalVerticalFlip, RandomBrightness, RandomGamma, AddGaussianNoise
from tqdm import tqdm


from sklearn import model_selection

from arguments import train_parser
from model.pomelo import JacobsUNet, PomeloUNet, ResBlocks, UResBlocks, ResBlocksDeep
from data.So2Sat import PopulationDataset_Reg
from utils.losses import get_loss, r2
from utils.metrics import get_test_metrics
from utils.utils import new_log, to_cuda, seed_all

from utils.utils import get_fnames_labs_reg, plot_2dmatrix, plot_and_save

from utils.constants import img_rows, img_cols, all_patches_mixed_train_part1, all_patches_mixed_test_part1

import wandb

import nvidia_smi
nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloaders = self.get_dataloaders(args)
        
        seed_all(args.seed)
        input_channels = args.Sentinel1*2 + args.Sentinel2*3 + args.VIIRS*1

        if args.model=="JacobsUNet":
            self.model = JacobsUNet( 
                input_channels = input_channels,
                feature_dim = args.feature_dim,
                feature_extractor = args.feature_extractor
            ).cuda()
        elif args.model=="PomeloUNet":
            self.model = PomeloUNet( 
                input_channels = input_channels,
                feature_dim = args.feature_dim,
                feature_extractor=args.feature_extractor
            ).cuda()
        elif args.model=="ResBlocks":
            self.model = ResBlocks(
                input_channels = input_channels,
                feature_dim = args.feature_dim,
            ).cuda()

        elif args.model=="UResBlocks":
            self.model = UResBlocks(
                input_channels = input_channels,
                feature_dim = args.feature_dim,
            ).cuda()

        elif args.model=="ResBlocksDeep":
            self.model = ResBlocksDeep(
                input_channels = input_channels,
                feature_dim = args.feature_dim,
            ).cuda()

        # number of params
        args.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Model", args.model, "; #Params:", args.pytorch_total_params)

        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, args.dataset), args)
        self.args.experiment_folder = self.experiment_folder

        wandb.init(project=args.wandb_project, dir=self.experiment_folder)
        wandb.config.update(self.args) 

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

        self.info = { "epoch": 0,  "iter": 0,  "sampleitr": 0}
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        if args.resume is not None:
            self.resume(path=args.resume)

    # def __del__(self):
    #     self.writer.close()

    def train(self):
        with tqdm(range(self.info["epoch"], self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.info["epoch"] + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()
                    self.test(plot=(( (self.info["epoch"])+1) % 4)==0)

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_gamma != 1.0: 
                    self.scheduler.step()
                    wandb.log({**{'log_lr': np.log10(self.scheduler.get_last_lr())}, **self.info}, self.info["iter"])

                self.info["epoch"] += 1

    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)

        self.model.train()
        
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        self.train_stats["gpu_used"] = info.used

        with tqdm(self.dataloaders['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            for i, sample in enumerate(inner_tnr):
                sample = to_cuda(sample)

                self.optimizer.zero_grad()

                output = self.model(sample, train=True)

                loss, loss_dict = get_loss(output, sample, loss=args.loss, lam=args.lam,
                                        merge_aug=args.merge_aug, lam_builtmask=args.lam_builtmask, lam_dense=args.lam_dense)

                if torch.isnan(loss):
                    raise Exception("detected NaN loss..")
                    
                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

                if self.info["epoch"] > 0 or not self.args.skip_first:
                    loss.backward()

                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                    self.optimizer.step()

                self.info["iter"] += 1
                self.info["sampleitr"] += self.args.batch_size

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                        validation_loss=self.val_stats['optimization_loss'],
                                        best_validation_loss=self.best_optimization_loss)

                    # upload logs
                    wandb.log({**{k + '/train': v for k, v in self.train_stats.items()}, **self.info}, self.info["iter"])
                    
                    # reset metrics
                    self.train_stats = defaultdict(float)

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            pred, gt = [], []
            for sample in tqdm(self.dataloaders["val"], leave=False):
                sample = to_cuda(sample)

                output = self.model(sample)
                
                # Colellect predictions and samples
                pred.append(output["popcount"].view(-1)); gt.append(sample["y"].view(-1))

                loss, loss_dict = get_loss(output, sample, loss=args.loss, lam=args.lam,
                                           merge_aug=args.merge_aug, lam_builtmask=args.lam_builtmask, lam_dense=args.lam_dense)

                for key in loss_dict:
                    self.val_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 
            
            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            # Compute non-averagable metrics
            self.val_stats["Population:r2"] = r2(torch.cat(pred), torch.cat(gt))

            wandb.log({**{k + '/val': v for k, v in self.val_stats.items()}, **self.info}, self.info["iter"])
            
            if self.val_stats['optimization_loss'] < self.best_optimization_loss:
                self.best_optimization_loss = self.val_stats['optimization_loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')

    def test(self, plot=False):
        self.test_stats = defaultdict(float)

        self.model.eval()
        cr_eval = False
        sum_pool10 = torch.nn.AvgPool2d(10, stride=10, divisor_override=1)
        sum_pool20 = torch.nn.AvgPool2d(20, stride=20, divisor_override=1)
        sum_pool40 = torch.nn.AvgPool2d(40, stride=40, divisor_override=1)
        sum_pool2 = torch.nn.AvgPool2d(2, stride=2, divisor_override=1)
        sum_pool4 = torch.nn.AvgPool2d(4, stride=4, divisor_override=1)

        s = 0
        pad = torch.ones(1, 100,100)

        with torch.no_grad():
            pred, gt = [], []
            pred2, gt2 = [], []
            pred4, gt4 = [], []
            pred10, gt10, gtSo2 = [], [], []
            for sample in tqdm(self.dataloaders["test"], leave=False):
                sample = to_cuda(sample)

                output = self.model(sample)
                
                # Colellect predictions and samples
                if cr_eval:
                    pred.append(output["popcount"].view(-1)); gt.append(sample["y"].view(-1)) 
                    loss, loss_dict = get_loss(output, sample, loss=args.loss, 
                                            merge_aug=args.merge_aug, lam_builtmask=args.lam_builtmask, lam_dense=args.lam_dense)

                    for key in loss_dict:
                        self.test_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                else:
                    #fine_eval
                    if sample["pop_avail"].any(): 
                        pred_zh = output["popdensemap"][sample["pop_avail"][:,0].bool()]
                        gt_zh = sample["Pop_X"][sample["pop_avail"][:,0].bool()]
                        PopNN_X = sample["PopNN_X"][sample["pop_avail"][:,0].bool()]
                        # gt_zhNN = sample["PopNN_X"][sample["pop_avail"][:,0].bool()]
                        # inputs_zh = sample["input"][sample["pop_avail"][:,0].bool()]

                        pred.append(sum_pool10(pred_zh).view(-1))
                        gt.append(gt_zh.view(-1))
                        pred2.append(sum_pool20(pred_zh).view(-1))
                        gt2.append(sum_pool2(gt_zh).view(-1))
                        pred4.append(sum_pool40(pred_zh).view(-1))
                        gt4.append(sum_pool4(gt_zh).view(-1))

                        gt10.append(sum_pool10(gt_zh).view(-1))
                        pred10.append(output["popcount"][sample["pop_avail"][:,0].bool()].view(-1))
                        gtSo2.append(sample["y"][sample["pop_avail"][:,0].bool()].view(-1))

                        i = 0
                        if plot==True:
                            for i in range(len(gt_zh)):
                                vmin, vmax = gt_zh[i].min(), gt_zh[i].max()
                                plot_and_save(gt_zh[i].cpu(), model_name=args.expN, title=gt_zh[i].sum().cpu().item(), vmin=vmin, vmax=vmax, idx=s, name="01_GT")
                                plot_and_save(sum_pool10(pred_zh)[i].cpu(), model_name=args.expN, title=sum_pool10(pred_zh)[i].sum().cpu().item(), vmin=vmin, vmax=vmax, idx=s, name="02_pred10")
                                plot_and_save(PopNN_X[i].cpu(), model_name=args.expN, title=(PopNN_X[i].sum()/100).cpu().item(), vmin=vmin, vmax=vmax, idx=s, name="03_GTNN")
                                plot_and_save(pred_zh[i].cpu()*100, model_name=args.expN, title=pred_zh[i].sum().cpu().item(), vmin=vmin, vmax=vmax, idx=s, name="04_pred100")

                                inp = sample["input"][sample["pop_avail"][:,0].bool()]
                                if args.Sentinel2:
                                    plot_and_save(inp[i,:3].cpu().permute(1,2,0)*0.2+0.5, model_name=args.expN, title=self.args.expN, idx=s, name="05_S2", cmap=None)
                                    if args.Sentinel1:
                                        plot_and_save(torch.cat([inp[i,3:5].cpu()*0.4 + 0.3, pad]).permute(1,2,0), model_name=args.expN, title=self.args.expN, idx=s, name="06_S1", cmap=None)
                                else:
                                    plot_and_save(torch.cat([inp[i,:2].cpu()*0.4 + 0.3, pad]).permute(1,2,0), model_name=args.expN, title=self.args.expN, idx=s, name="06_S1", cmap=None)

                                s += 1
                            plot = s<100

            if cr_eval:
                self.test_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}
                # Compute non-averagable metrics
                self.test_stats["Population:r2"] = r2(torch.cat(pred), torch.cat(gt))
                wandb.log({**{k + '/test': v for k, v in self.test_stats.items()}, **self.info}, self.info["iter"])

            else:
                self.test_stats1 = get_test_metrics(torch.cat(pred), torch.cat(gt), tag="100m")
                self.test_stats2 = get_test_metrics(torch.cat(pred2), torch.cat(gt2), tag="200m")
                self.test_stats4 = get_test_metrics(torch.cat(pred4), torch.cat(gt4), tag="400m")
                self.test_stats10 = get_test_metrics(torch.cat(pred10), torch.cat(gt10), tag="1km")
                self.test_statsGT = get_test_metrics(torch.cat(gt10), torch.cat(gtSo2), tag="GTCons")
                self.test_stats = {**self.test_stats1, **self.test_stats2, **self.test_stats4, **self.test_stats10, **self.test_statsGT}
                wandb.log({**{k + '/testZH': v for k, v in self.test_stats.items()}, **self.info}, self.info["iter"])


            

    @staticmethod
    def get_dataloaders(args): 

        phases = ('train', 'val')
        if args.dataset == 'So2Sat':
            params = {'dim': (img_rows, img_cols), "satmode": args.satmode}

            if not args.Sentinel1: 
                data_transform = transforms.Compose([
                    AddGaussianNoise(std=0.1, p=0.9),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomVerticalFlip(p=0.5),
                    # RandomRotationTransform(angles=[90, 180, 270], p=0.75),
                    # RandomGamma(),
                    # RandomBrightness()
                ])
            else:
                data_transform = transforms.Compose([
                    # RandomHorizontalVerticalFlip(p=0.5),
                    AddGaussianNoise(std=0.1, p=0.9),
                    # transforms.RandomVerticalFlip(p=0.5),
                    # RandomRotationTransform(angles=[90, 180, 270], p=0.75),
                    # RandomGamma(),
                    # RandomBrightness()
                ])
                
            val_size = 0.2 
            f_names, labels = get_fnames_labs_reg(all_patches_mixed_train_part1, force_recompute=False)
            f_names, labels = f_names[:int(args.max_samples)] , labels[:int(args.max_samples)] 
            # f_names_train, f_names_val, labels_train, labels_val = model_selection.train_test_split(
            #     f_names, labels, test_size=val_size, random_state=42)
            s = int(len(f_names)*val_size)
            f_names_train, f_names_val, labels_train, labels_val = f_names[:-s], f_names[-s:], labels[:-s], labels[-s:]

            f_names_test, labels_test = get_fnames_labs_reg(all_patches_mixed_test_part1, force_recompute=False)

            datasets = {
                "train": PopulationDataset_Reg(f_names_train, labels_train, mode="train", in_memory=args.in_memory,
                                                S1=args.Sentinel1, S2=args.Sentinel2, VIIRS=args.VIIRS,  transform=data_transform, **params),
                "val": PopulationDataset_Reg(f_names_val, labels_val, mode="val", in_memory=args.in_memory, 
                                                S1=args.Sentinel1, S2=args.Sentinel2, VIIRS=args.VIIRS,  transform=None, **params),
                "test": PopulationDataset_Reg(f_names_test, labels_test, mode="test", in_memory=args.in_memory, 
                                                S1=args.Sentinel1, S2=args.Sentinel2, VIIRS=args.VIIRS,  transform=None, **params)
            }
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')
        
        return {"train": DataLoader(datasets["train"], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True),
                "val":  DataLoader(datasets["val"], batch_size=args.batch_size*4, num_workers=args.num_workers, shuffle=False, drop_last=False),
                "test":  DataLoader(datasets["test"], batch_size=args.batch_size*4, num_workers=args.num_workers, shuffle=False, drop_last=False)}

    def save_model(self, prefix=''): 
        torch.save({
            'model': self.model.state_dict(),
            'epoch': self.info["epoch"] + 1,
            'iter': self.info["iter"]
        }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if not args.no_opt:
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
