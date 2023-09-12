

import argparse
from torch.utils.data import DataLoader
from data.PopulationDataset_target import Population_Dataset_target
# from model.DDA_model.utils import networks #,datasets, loss_functions, evaluation, experiment_manager, parsers
from model.DDA_model.utils.networks import load_checkpoint

from utils.utils import Namespace
import torch
from tqdm import tqdm
from rasterio.windows import Window
import numpy as np

from utils.utils import to_cuda_inplace

import rasterio
import os
from shutil import copyfile
from utils.utils import load_json, apply_transformations_and_normalize, apply_normalize
from utils.constants import config_path

from utils.plot import plot_2dmatrix

from utils.constants import pop_map_root_large, pop_map_root


# chunk size
chunk_size = 1000

def main(args):

    # get dataset
    overlap = 128
    # ips = 512
    # ips = 1024
    # ips = 4096
    ips = 2048
    
    input_defs = {'S1': True, 'S2': True, 'VIIRS': False, 'NIR': True}
    dataset = Population_Dataset_target(args.region, patchsize=ips, overlap=overlap, fourseasons=False,
                                        sentinelbuildings=False, ascfill=False, **input_defs)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

    # get model
    MODEL = Namespace(TYPE='dualstreamunet', OUT_CHANNELS=1, IN_CHANNELS=6, TOPOLOGY=[64, 128,] )
    # CONSISTENCY_TRAINER = Namespace(LOSS_FACTOR=0.0)
    CONSISTENCY_TRAINER = Namespace(LOSS_FACTOR=0.5)
    # PATHS = Namespace(OUTPUT="/scratch2/metzgern/HAC/data/DDAdata/outputs")
    PATHS = Namespace(OUTPUT="/scratch2/metzgern/HAC/data/DDAdata/outputsDDA")
    DATALOADER = Namespace(SENTINEL1_BANDS=['VV', 'VH'], SENTINEL2_BANDS=['B02', 'B03', 'B04', 'B08'])
    TRAINER = Namespace(LR=1e5)
    cfg = Namespace(MODEL=MODEL, CONSISTENCY_TRAINER=CONSISTENCY_TRAINER, PATHS=PATHS,
                    DATALOADER=DATALOADER, TRAINER=TRAINER, NAME="fusionda_new")

    ## load weights from checkpoint
    net, _, _ = load_checkpoint(epoch=15, cfg=cfg, device="cuda", no_disc=True)

    # get dataset stats
    dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats_unified_2A.json'))
    for mkey in dataset_stats.keys():
        if isinstance(dataset_stats[mkey], dict):
            for key,val in dataset_stats[mkey].items():
                dataset_stats[mkey][key] = torch.tensor(val)
        else:
            dataset_stats[mkey] = torch.tensor(val)

    # set to model to eval mode
    net.eval()

    # get predictions
    with torch.no_grad(): 
        h, w = dataloader.dataset.shape()
        # output_map_count = torch.zeros((h, w), dtype=torch.int8)
        # output_map = torch.zeros((h, w), dtype=torch.float16)

        # save predictions to file
        global pop_map_root
        pop_map_root = pop_map_root.replace("scratch", "scratch3")
        region_root = os.path.join(pop_map_root, args.region)
        metadata1 = dataloader.dataset.metadata()
        metadata1.update({"count": 1,
                        #  "dtype": "float32",
                        "dtype": "int32",
                        "compress": "PACKBITS"
                        })
    

        tmp_output_map_file = os.path.join(region_root, "tmp_output_map.tif")
        tmp_output_map_count_file = os.path.join(region_root, "tmp_output_map_count.tif")

        # initialize temporary files with zeros
        print("Initializing temporary files with zeros...")
        # Decide on a suitable chunk size (can be adjusted for your needs)
        chunk_size = 4096  # or any other reasonable value

        # Initialize temporary raster files and write zeros to them in chunks
        with rasterio.open(tmp_output_map_file, 'w', **metadata1) as tmp_dst:
            # rasterio.open(tmp_output_map_count_file, 'w', **metadata1) as tmp_count_dst:
            
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

        # Initialize temporary raster files
        with rasterio.open(tmp_output_map_file, 'r+', **metadata1) as tmp_dst, \
            rasterio.open(tmp_output_map_count_file, 'r+', **metadata1) as tmp_count_dst:
            
            for i, sample in tqdm(enumerate(dataloader), leave=True, total=len(dataloader)):

                sample = to_cuda_inplace(sample)
                # sample = apply_normalize(sample, self.dataset_stats)
                sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=dataset_stats)

                # get the format right reverese the first 3 channels of S2
                S1 = sample["input"][:, 4:6] 
                S2_RGB = torch.flip(sample["input"][:, :3],dims=(1,))
                S2_NIR = sample["input"][:, 3:4]
                x_fusion = torch.cat([S1, S2_RGB, S2_NIR], dim=1)

                xl,yl = [val.item() for val in sample["img_coords"]]
                mask = sample["mask"][0].bool()

                _, _, fusion_logits, _, _ = net(x_fusion, alpha=0)

                fusion_logits = fusion_logits.squeeze(0).squeeze(0)
                
                # Apply sigmoid activation
                fusion_logits = torch.sigmoid(fusion_logits)
    
                # Save current predictions to temporary file
                xl, yl, xu, yu = xl, yl, xl+ips, yl+ips
                window = Window(yl, xl, yu-yl, xu-xl)

                # Read existing values, sum new values (accounting for mask), and write back
                existing_values = tmp_dst.read(1, window=window).astype(np.int32)
                existing_values[mask.cpu().numpy()] += (fusion_logits[mask].cpu().numpy()*255).astype(np.int32)
                tmp_dst.write(existing_values, 1, window=window)
                
                # test
                # a = tmp_dst.read(1, window=window).astype(np.int32)
                
                # Increment count in the count map
                output_map_count = tmp_count_dst.read(1, window=window).astype(np.int32)
                output_map_count[mask.cpu().numpy()] += 1
                # count_chunk = np.ones(fusion_logits.shape, dtype=np.int8)
                tmp_count_dst.write(output_map_count, 1, window=window)

                # if i==10:
                #     break

    # average predictions
    # div_mask = output_map_count > 1
    # output_map[div_mask] = output_map[div_mask] / output_map_count[div_mask]

    # del output_map_count
    del sample

    # save predictions to file
    region_root = os.path.join(pop_map_root, args.region)
    metadata = dataloader.dataset.metadata()
    metadata.update({"count": 1,
                    #  "dtype": "float32",
                     "dtype": "uint8",
                     "compress": "PACKBITS"
                     })

    # Read the temporary maps in chunks, average and write to the final output map
    with rasterio.open(tmp_output_map_file, 'r') as tmp_src, \
        rasterio.open(tmp_output_map_count_file, 'r') as tmp_count_src, \
        rasterio.open(os.path.join(region_root, "buildingsDDA2_44C.tif"), 'w', **metadata) as dst:

        for i in tqdm(range(0, h, chunk_size)):
            for j in tqdm(range(0, w, chunk_size), leave=False):
                # Adjust the shape of the chunk for edge cases
                chunk_height = min(chunk_size, h - i)
                chunk_width = min(chunk_size, w - j)
                
                # Read chunks
                window = Window(j, i, chunk_width, chunk_height)
                    
                data_chunk = tmp_src.read(1, window=window)
                count_chunk = tmp_count_src.read(1, window=window)
                    
                # Average the data chunk
                div_mask_chunk = count_chunk > 1
                data_chunk[div_mask_chunk] = data_chunk[div_mask_chunk] / count_chunk[div_mask_chunk]
                    
                # Write the chunk to the final file
                dst.write((data_chunk).astype(np.uint8), 1, window=window)

                
    # Optionally delete the temporary files after the operation
    # os.remove(tmp_output_map_file)
    # os.remove(tmp_output_map_count_file)


    return None



if __name__=="__main__":


    # intialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg', '--region', type=str, help='')
    args = parser.parse_args()

    # run main
    main(args)
    print("Done!")