import os
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
from torch import is_tensor, optim
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision import transforms
from utils.transform import RandomRotationTransform, RandomBrightness, RandomGamma
from tqdm import tqdm

from sklearn import model_selection

from arguments import train_parser
from model.pomelo import JacobsUNet
from data.So2Sat import PopulationDataset_Reg
from utils.losses import get_loss
from utils.utils import new_log, to_cuda, seed_all

from utils.utils import get_fnames_labs_reg

from utils.constants import img_rows, img_cols, all_patches_mixed_train_part1

import wandb

import nvidia_smi
nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloaders = self.get_dataloaders(args)
        
        seed_all(args.seed)

        self.model = JacobsUNet( 
            input_channels = 4,
            feature_dim = 32,
        ).cuda()

        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, args.dataset), args)
        self.args.experiment_folder = self.experiment_folder

        wandb.init(project=args.wandb_project, dir=self.experiment_folder)
        wandb.config.update(self.args)
        # self.writer = None

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        if args.resume is not None:
            self.resume(path=args.resume)

    # def __del__(self):
    #     self.writer.close()

    def train(self):
        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_gamma != 1.0: 
                    self.scheduler.step()
                    wandb.log({'log_lr': np.log10(self.scheduler.get_last_lr())}, self.iter)

                self.epoch += 1

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

                loss, loss_dict = get_loss(output, sample)

                if torch.isnan(loss):
                    raise Exception("detected NaN loss..")
                    
                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

                if self.epoch > 0 or not self.args.skip_first:
                    loss.backward()

                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                    self.optimizer.step()

                self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                        validation_loss=self.val_stats['optimization_loss'],
                                        best_validation_loss=self.best_optimization_loss)

                    # upload logs
                    wandb.log({k + '/train': v for k, v in self.train_stats.items()}, self.iter)
                    
                    # reset metrics
                    self.train_stats = defaultdict(float)

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloaders['val'], leave=False):
                sample = to_cuda(sample)

                output = self.model(sample)

                loss, loss_dict = get_loss(output, sample)

                for key in loss_dict:
                    self.val_stats[key] +=  loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            wandb.log({k + '/val': v for k, v in self.val_stats.items()}, self.iter)
            
            if self.val_stats['optimization_loss'] < self.best_optimization_loss:
                self.best_optimization_loss = self.val_stats['optimization_loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')

    @staticmethod
    def get_dataloaders(args): 

        phases = ('train', 'val')
        if args.dataset == 'So2Sat':
            params = {'dim': (img_rows, img_cols), "satmode": args.satmode}
            data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotationTransform(angles=[90, 180, 270], p=0.5),
            RandomGamma(),
            RandomBrightness()
        ])
            
            val_size = 0.2
            data_dir = all_patches_mixed_train_part1
            f_names, labels = get_fnames_labs_reg(data_dir)
            f_names_train, f_names_val, labels_train, labels_val = model_selection.train_test_split(
             f_names, labels, test_size=val_size, random_state=42)

            datasets = {
                "train": PopulationDataset_Reg(f_names_train, labels_train, transform=data_transform, **params),
                "val": PopulationDataset_Reg(f_names_val, labels_val, **params)
            }
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return {phase: DataLoader(datasets[phase], batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=True, drop_last=False) for phase in phases}

    def save_model(self, prefix=''):
        if args.no_opt:
            torch.save({
                'model': self.model.state_dict(),
                'epoch': self.epoch + 1,
                'iter': self.iter
            }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))
        else:
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch + 1,
                'iter': self.iter
            }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if not args.no_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = train_parser.parse_args()
    print(train_parser.format_values())


    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
