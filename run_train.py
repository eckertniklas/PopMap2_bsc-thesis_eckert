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
from data.PopulationDataset_target import Population_Dataset_target
from utils.losses import get_loss, r2
from utils.metrics import get_test_metrics
from utils.utils import new_log, to_cuda, seed_all

from utils.utils import get_fnames_labs_reg, get_fnames_unlab_reg, plot_2dmatrix, plot_and_save
from utils.datasampler import LabeledUnlabeledSampler
from utils.constants import img_rows, img_cols, all_patches_mixed_train_part1, all_patches_mixed_test_part1, pop_map_root, inference_patch_size, overlap
from utils.constants import inference_patch_size as ips
import wandb

import nvidia_smi
nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # set up dataloaders
        self.dataloaders = self.get_dataloaders(args)
        
        # set up model
        seed_all(args.seed)

        # define input channels based on the number of input modalities
        input_channels = args.Sentinel1*2 + args.Sentinel2*3 + args.VIIRS*1

        # define architecture
        if args.model=="JacobsUNet":
            self.model = JacobsUNet( 
                input_channels = input_channels,
                feature_dim = args.feature_dim,
                feature_extractor = args.feature_extractor,
                classifier = args.classifier
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

        # set up experiment folder
        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, args.dataset), args)
        self.args.experiment_folder = self.experiment_folder

        # wandb config
        wandb.init(project=args.wandb_project, dir=self.experiment_folder)
        wandb.config.update(self.args) 

        # set up optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

        # set up info
        self.info = { "epoch": 0,  "iter": 0,  "sampleitr": 0}
        self.info["alpha"] = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        # in case of checkpoint resume
        if args.resume is not None:
            self.resume(path=args.resume)


    def train(self):
        with tqdm(range(self.info["epoch"], self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                # in domain validation
                if (self.info["epoch"] + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()
                    torch.cuda.empty_cache()
                
                # target domain testing
                if (self.info["epoch"] + 1) % 5 == 0:
                    # self.test(plot=((self.info["epoch"]+1) % 4)==0, full_eval=((self.info["epoch"]+1) % 1)==0, zh_eval=True) 
                    self.test(plot=((self.info["epoch"]+1) % 1)==0, full_eval=True, zh_eval=True) #ZH
                    torch.cuda.empty_cache()
                if (self.info["epoch"] + 1) % 1 == 0:
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
        self.train_stats = defaultdict(float)

        # set model to train mode
        self.model.train()

        # get GPU memory usage
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        self.train_stats["gpu_used"] = info.used

        with tqdm(self.dataloaders['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            for i, sample in enumerate(inner_tnr):
                self.optimizer.zero_grad()

                # forward pass
                sample = to_cuda(sample)
                output = self.model(sample, train=True, alpha=self.info["alpha"] if self.args.adversarial else 0.)
                
                # compute loss
                loss, loss_dict = get_loss(output, sample, loss=args.loss, lam=args.lam, merge_aug=args.merge_aug,
                                           lam_builtmask=args.lam_builtmask, lam_dense=args.lam_dense, lam_adv=args.lam_adv if self.args.adversarial else 0.0)

                # detect NaN loss 
                if torch.isnan(loss):
                    raise Exception("detected NaN loss..")

                # accumulate stats 
                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

                # backprop
                if self.info["epoch"] > 0 or not self.args.skip_first:
                    loss.backward()

                    # gradient clipping
                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                    self.optimizer.step()
                
                # update info
                self.info["iter"] += 1
                self.info["sampleitr"] += self.args.batch_size//2 if self.args.adversarial else self.args.batch_size

                # update alpha for the adversarial loss
                if self.args.adversarial:
                    # an annealing (to 1.0) schedule for alpha
                    p = float(i + self.info["epoch"] * len(self.dataloaders['train'])) / self.args.num_epochs / len(self.dataloaders['train'])
                    self.info["alpha"] = 2. / (1. + np.exp(-10 * p)) - 1

                # logging
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

                # forward pass
                sample = to_cuda(sample)
                output = self.model(sample)
                
                # Colellect predictions and samples
                pred.append(output["popcount"].view(-1)); gt.append(sample["y"].view(-1))
                
                # compute loss
                loss, loss_dict = get_loss(output, sample, loss=args.loss, lam=args.lam, merge_aug=args.merge_aug,
                                           lam_builtmask=args.lam_builtmask, lam_dense=args.lam_dense, lam_adv=args.lam_adv if self.args.adversarial else 0.0)

                # accumulate stats
                for key in loss_dict:
                    self.val_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 
            
            # Compute average metrics
            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            # Compute non-averagable metrics
            self.val_stats["Population:r2"] = r2(torch.cat(pred), torch.cat(gt))

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
            pred2, gt2 = [], []
            pred4, gt4 = [], []
            pred10, gt10, gtSo2 = [], [], []
            for sample in tqdm(self.dataloaders["test"], leave=False):

                # forward pass
                sample = to_cuda(sample)
                output = self.model(sample)
                
                # Colellect predictions and samples
                if full_eval:
                    pred.append(output["popcount"].view(-1)); gt.append(sample["y"].view(-1)) 
                    loss, loss_dict = get_loss(output, sample, loss=args.loss, merge_aug=args.merge_aug,
                                               lam_builtmask=args.lam_builtmask, lam_dense=args.lam_dense, lam_adv=args.lam_adv if self.args.adversarial else 0.0)

                    for key in loss_dict:
                        self.test_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                
                #fine_eval for Zurich
                if zh_eval:
                    if sample["pop_avail"].any(): 
                        pred_zh = output["popdensemap"][sample["pop_avail"][:,0].bool()]
                        gt_zh = sample["Pop_X"][sample["pop_avail"][:,0].bool()]
                        PopNN_X = sample["PopNN_X"][sample["pop_avail"][:,0].bool()]

                        # Collect all different aggregation scales  (1, 2, 4, 10)
                        pred.append(sum_pool10(pred_zh).view(-1))
                        gt.append(gt_zh.view(-1))
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
                            for i in tqdm(range(len(gt_zh))):
                                vmax = max([gt_zh[i].max(), pred_zh[i].max()*100]).cpu().item()
                                plot_and_save(gt_zh[i].cpu(), model_name=args.expN, title=gt_zh[i].sum().cpu().item(), vmin=0, vmax=vmax, idx=s, name="01_GT", folder=self.args.experiment_folder)
                                plot_and_save(sum_pool10(pred_zh)[i].cpu(), model_name=args.expN, title=sum_pool10(pred_zh)[i].sum().cpu().item(), vmin=0, vmax=vmax, idx=s, name="02_pred10", folder=self.args.experiment_folder)
                                plot_and_save(PopNN_X[i].cpu(), model_name=args.expN, title=(PopNN_X[i].sum()/100).cpu().item(), vmin=0, vmax=vmax, idx=s, name="03_GTNN", folder=self.args.experiment_folder)
                                plot_and_save(pred_zh[i].cpu()*100, model_name=args.expN, title=pred_zh[i].sum().cpu().item(), vmin=0, vmax=vmax, idx=s, name="04_pred100", folder=self.args.experiment_folder)

                                inp = sample["input"][sample["pop_avail"][:,0].bool()]
                                if args.Sentinel2:
                                    plot_and_save(inp[i,:3].cpu().permute(1,2,0)*0.2+0.5, model_name=args.expN, title=self.args.expN, idx=s, name="05_S2", cmap=None, folder=self.args.experiment_folder)
                                    if args.Sentinel1:
                                        plot_and_save(torch.cat([inp[i,3:5].cpu()*0.4 + 0.3, pad]).permute(1,2,0), model_name=args.expN, title=self.args.expN, idx=s, name="06_S1", cmap=None, folder=self.args.experiment_folder)
                                else:
                                    plot_and_save(torch.cat([inp[i,:2].cpu()*0.4 + 0.3, pad]).permute(1,2,0), model_name=args.expN, title=self.args.expN, idx=s, name="06_S1", cmap=None, folder=self.args.experiment_folder)

                                s += 1
                                if s > 300:
                                    break

            if full_eval:
                # average all stats
                self.test_stats = {k: v / len(self.dataloaders['val']) for k, v in self.test_stats.items()}

                # Compute non-averagable metrics
                self.test_stats["Population:r2"] = r2(torch.cat(pred), torch.cat(gt))
                wandb.log({**{k + '/test': v for k, v in self.test_stats.items()}, **self.info}, self.info["iter"])

            if zh_eval:
                self.test_stats1 = get_test_metrics(torch.cat(pred), torch.cat(gt), tag="100m")
                self.test_stats2 = get_test_metrics(torch.cat(pred2), torch.cat(gt2), tag="200m")
                self.test_stats4 = get_test_metrics(torch.cat(pred4), torch.cat(gt4), tag="400m")
                self.test_stats10 = get_test_metrics(torch.cat(pred10), torch.cat(gt10), tag="1km")
                self.test_statsGT = get_test_metrics(torch.cat(gt10), torch.cat(gtSo2), tag="GTCons")
                self.test_stats = {**self.test_stats, **self.test_stats1, **self.test_stats2, **self.test_stats4, **self.test_stats10, **self.test_statsGT}
            
                wandb.log({**{k + '/testZH': v for k, v in self.test_stats.items()}, **self.info}, self.info["iter"])


    def test_target(self, save=False):
        # Test on target domain
        self.model.eval()
        self.test_stats = defaultdict(float)

        with torch.no_grad():  
            # test_dataloader = True
            # if test_dataloader:
            #     for i in range(3):
            #         sample = self.dataloaders["test_target"][0].dataset[i]

            for testdataloader in self.dataloaders["test_target"]:
                    
                # inputialize the output map
                h, w = testdataloader.dataset.shape()
                output_map = torch.zeros((h, w))
                output_map_count = torch.zeros((h, w))

                for sample in tqdm(testdataloader, leave=False):
                    
                    sample = to_cuda(sample)
                    xmin, xmax, ymin, ymax = [val.item() for val in sample["valid_coords"]]
                    xl,yl = [val.item() for val in sample["img_coords"]]
                    mask = sample["mask"][0].bool()

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
    def get_dataloaders(args): 

        phases = ('train', 'val')
        if args.dataset == 'So2Sat':
            params = {'dim': (img_rows, img_cols), "satmode": args.satmode, 'in_memory': args.in_memory,
                      'S1': args.Sentinel1, 'S2': args.Sentinel2, 'VIIRS': args.VIIRS}

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
            
            # source domain samples
            val_size = 0.2 
            f_names, labels = get_fnames_labs_reg(all_patches_mixed_train_part1, force_recompute=False)
            f_names, labels = f_names[:int(args.max_samples)] , labels[:int(args.max_samples)]
            s = int(len(f_names)*val_size)
            f_names_train, f_names_val, labels_train, labels_val = f_names[:-s], f_names[-s:], labels[:-s], labels[-s:]
            f_names_test, labels_test = get_fnames_labs_reg(all_patches_mixed_test_part1, force_recompute=False)

            # target domain samples
            f_names_unlab = []
            if args.adversarial:
                for reg in args.target_regions:
                    this_unlabeled_path = os.path.join(pop_map_root, os.path.join("EE", reg))
                    f_names_unlab.extend(get_fnames_unlab_reg(this_unlabeled_path, force_recompute=False)) 

            datasets = {
                "train": PopulationDataset_Reg(f_names_train, labels_train, f_names_unlab=f_names_unlab, mode="train", transform=data_transform,random_season=args.random_season, **params),
                "val": PopulationDataset_Reg(f_names_val, labels_val, mode="val", transform=None, **params),
                "test": PopulationDataset_Reg(f_names_test, labels_test, mode="test", transform=None, **params),
                "test_target": [ Population_Dataset_target(reg, S1=args.Sentinel1, S2= args.Sentinel2, VIIRS=args.VIIRS,
                                                           patchsize=ips, overlap=overlap) for reg in args.target_regions ]
            }
            
            if len(args.target_regions)>0 and len(datasets["train"].unlabeled_indices)>0:
                custom_sampler = LabeledUnlabeledSampler(
                    labeled_indices=datasets["train"].labeled_indices,
                    unlabeled_indices=datasets["train"].unlabeled_indices,
                    batch_size=args.batch_size
                )
                shuffle = False
            else:
                custom_sampler = None
                shuffle = True

        else:
            raise NotImplementedError(f'Dataset {args.dataset}')
        
        return {"train": DataLoader(datasets["train"], batch_size=args.batch_size, num_workers=args.num_workers, sampler=custom_sampler, shuffle=shuffle, drop_last=True),
                "val":  DataLoader(datasets["val"], batch_size=args.batch_size*2, num_workers=args.num_workers, shuffle=False, drop_last=False),
                "test":  DataLoader(datasets["test"], batch_size=args.batch_size*2, num_workers=args.num_workers, shuffle=False, drop_last=False),
                "test_target":  [DataLoader(datasets["test_target"], batch_size=1, num_workers=1, shuffle=False, drop_last=False) for datasets["test_target"] in datasets["test_target"] ]
                }

    def save_model(self, prefix=''):
        torch.save({
            'model': self.model.state_dict(),
            'epoch': self.info["epoch"] + 1,
            'iter': self.info["iter"],
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

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
