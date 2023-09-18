import os
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
# from torch import is_tensor, optim
# from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ChainDataset, ConcatDataset
# from torchvision.transforms import Normalize
# from torchvision import transforms
# from utils.transform import OwnCompose
from utils.transform import RandomRotationTransform, RandomHorizontalFlip, RandomVerticalFlip, RandomHorizontalVerticalFlip, RandomBrightness, RandomGamma, HazeAdditionModule, AddGaussianNoise
from tqdm import tqdm

# import itertools
# import random
from sklearn import model_selection
import wandb

# import pickle
# import gc

import rasterio
from rasterio.windows import Window
from shutil import copyfile

# from arguments import eval_parser
from arguments.eval import parser as eval_parser
# from data.So2Sat import PopulationDataset_Reg
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

        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder)

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


    def test_target(self, save=False, full=True, save_scatter=False):

        # if afganistan or uganda is in the test set, we need to use the large test function
        if any([el in self.args.target_regions for el in ["afg", "uga"]]):
            self.test_target_large(save=save, full=full)
            return
        
        # Test on target domain
        save_scatter = save_scatter
        self.model.eval()
        self.test_stats = defaultdict(float)
        # self.model.train()

        with torch.no_grad(): 
            self.target_test_stats = defaultdict(float)
            for testdataloader in self.dataloaders["test_target"]: 

                # inputialize the output map
                h, w = testdataloader.dataset.shape()
                output_map = torch.zeros((h, w), dtype=torch.float16)
                output_scale_map = torch.zeros((h, w), dtype=torch.float16)
                output_map_count = torch.zeros((h, w), dtype=torch.int8)

                for sample in tqdm(testdataloader, leave=True):
                    sample = to_cuda_inplace(sample)
                    # sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput,
                    #     segmentationinput=self.args.segmentationinput)
                    sample = apply_transformations_and_normalize(sample,  transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput,
                                                                      segmentationinput=self.args.segmentationinput, empty_eps=self.args.empty_eps)

                    # get the valid coordinates
                    xl,yl = [val.item() for val in sample["img_coords"]]
                    mask = sample["mask"][0].bool()

                    # get the output with a forward pass
                    output = self.model(sample, padding=False)
                    output_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["popdensemap"][0][mask].cpu().to(torch.float16)

                    if "scale" in output.keys():
                        output_scale_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["scale"][0][mask].cpu().to(torch.float16)

                    output_map_count[xl:xl+ips, yl:yl+ips][mask.cpu()] += 1

                ###### average over the number of times each pixel was visited ######
                # mask out values that are not visited of visited exactly once
                div_mask = output_map_count > 1
                output_map[div_mask] = output_map[div_mask] / output_map_count[div_mask]

                if "scale" in output.keys():
                    output_scale_map[div_mask] = output_scale_map[div_mask] / output_map_count[div_mask]
                
                # save maps
                print("saving maps")
                if save:
                    # save the output map
                    testdataloader.dataset.save(output_map, self.experiment_folder)

                    if "scale" in output.keys():
                        testdataloader.dataset.save(output_scale_map, self.experiment_folder, tag="SCALE_{}".format(testdataloader.dataset.region))
                
                # convert populationmap to census
                gpu_mode = True
                for level in testlevels[testdataloader.dataset.region]:
                    print("-"*50)
                    print("Evaluating level: ", level)
                    # convert map to census
                    census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map, gpu_mode=gpu_mode, level=level, details_to=os.path.join(self.experiment_folder, "{}_{}".format(testdataloader.dataset.region, level)))
                    this_metrics = get_test_metrics(census_pred, census_gt.float().cuda(), tag="MainCensus_{}_{}".format(testdataloader.dataset.region, level))
                    print(this_metrics)
                    self.target_test_stats = {**self.target_test_stats, **this_metrics}

                    # get the metrics for the clearly built up areas
                    built_up = census_gt>10
                    self.target_test_stats = {**self.target_test_stats,
                                              **get_test_metrics(census_pred[built_up], census_gt[built_up].float().cuda(), tag="MainCensusPos_{}_{}".format(testdataloader.dataset.region, level))}
                    
                    # create scatterplot and upload to wandb
                    # print(self.target_test_stats)
                    if save_scatter:
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
                    census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map_adj, gpu_mode=gpu_mode, level=level, details_to=os.path.join(self.experiment_folder, "{}_{}_adj".format(testdataloader.dataset.region, level)))
                    test_stats_adj = get_test_metrics(census_pred, census_gt.float().cuda(), tag="AdjCensus_{}_{}".format(testdataloader.dataset.region, level))
                    print(test_stats_adj)
                    
                    built_up = census_gt>10
                    test_stats_adj = {**test_stats_adj,
                                      **get_test_metrics(census_pred[built_up], census_gt[built_up].float().cuda(), tag="AdjCensusPos_{}_{}".format(testdataloader.dataset.region, level))}
                    
                    # print(test_stats_adj)
                    self.target_test_stats = {**self.target_test_stats,
                                              **test_stats_adj}

                    if save_scatter:
                        # create scatterplot and upload to wandb
                        scatterplot = scatter_plot3(census_pred.tolist(), census_gt.tolist(), log_scale=True)
                        if scatterplot is not None:
                            self.target_test_stats["Scatter/Scatter_{}_{}_adj".format(testdataloader.dataset.region, level)] = wandb.Image(scatterplot)
                    
            
            # save the target test stats
            wandb.log({**{k + '/targettest': v for k, v in self.target_test_stats.items()}, **self.info}, self.info["iter"])



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

                                        # data_chunk[div_mask_chunk] = data_chunk[div_mask_chunk] / count_chunk[div_mask_chunk]
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
                                    # data_chunk[div_mask_chunk] /= count_chunk[div_mask_chunk]
                                    
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
        need_asc = ["uga"]
        datasets = {
            "test_target": [ Population_Dataset_target(reg, patchsize=ips, overlap=overlap, sentinelbuildings=args.sentinelbuildings, ascfill=reg in need_asc,
                                                       fourseasons=self.args.fourseasons, train_level=lvl, **input_defs)
                                for reg,lvl in zip(args.target_regions, args.train_level) ]
        }
        
        # create the dataloaders
        dataloaders =  {
            "test_target":  [DataLoader(datasets["test_target"], batch_size=1, num_workers=8, shuffle=False, drop_last=False)
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
