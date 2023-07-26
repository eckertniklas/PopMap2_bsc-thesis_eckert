

import argparse
from torch.utils.data import DataLoader
from data.PopulationDataset_target import Population_Dataset_target
from model.DDA_model.utils import networks#,datasets, loss_functions, evaluation, experiment_manager, parsers
from model.DDA_model.utils.networks import load_checkpoint

from utils.utils import Namespace
import torch
from tqdm import tqdm

from utils.utils import to_cuda_inplace

import rasterio
import os

from utils.utils import load_json, apply_transformations_and_normalize, apply_normalize
from utils.constants import config_path

from utils.plot import plot_2dmatrix

from utils.constants import pop_map_root_large, pop_map_root

def main(args):

    # get dataset
    overlap = 128
    ips = 512
    # ips = 1024
    # ips = 4096
    # ips = 2048
    
    input_defs = {'S1': True, 'S2': True, 'VIIRS': False, 'NIR': True}
    dataset = Population_Dataset_target(args.region, patchsize=ips, overlap=overlap, fourseasons=True,
                                        sentinelbuildings=True, ascfill=True, **input_defs)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

    # get model
    MODEL = Namespace(TYPE='dualstreamunet', OUT_CHANNELS=1, IN_CHANNELS=6, TOPOLOGY=[64, 128,] )
    CONSISTENCY_TRAINER = Namespace(LOSS_FACTOR=0.0)
    PATHS = Namespace(OUTPUT="/scratch2/metzgern/HAC/data/DDAdata/outputs")
    DATALOADER = Namespace(SENTINEL1_BANDS=['VV', 'VH'], SENTINEL2_BANDS=['B02', 'B03', 'B04', 'B08'])
    TRAINER = Namespace(LR=1e5)
    cfg = Namespace(MODEL=MODEL, CONSISTENCY_TRAINER=CONSISTENCY_TRAINER, PATHS=PATHS,
                    DATALOADER=DATALOADER, TRAINER=TRAINER, NAME="fusionda_new")

    ## load weights from checkpoint
    net, _, _ = load_checkpoint(epoch=15, cfg=cfg, device="cuda")

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
        output_map_count = torch.zeros((h, w), dtype=torch.int8)
        output_map = torch.zeros((h, w), dtype=torch.float16) 
            
        for sample in tqdm(dataloader, leave=True):

            sample = to_cuda_inplace(sample)
            # sample = apply_normalize(sample, self.dataset_stats)
            sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=dataset_stats)

            # get the format right
            # reverese the first 3 channels?
            # S2 = sample["input"][:, :4]
            S1 = sample["input"][:, 4:6] 
            S2_RGB = torch.flip(sample["input"][:, :3],dims=(1,))
            S2_NIR = sample["input"][:, 3:4]

            x_fusion = torch.cat([S1, S2_RGB, S2_NIR], dim=1)

            xl,yl = [val.item() for val in sample["img_coords"]]
            mask = sample["mask"][0].bool()

            # forward pass
            # sar_logits, optical_logits, fusion_logits, disc_logits_sar, disc_logits_optical = net(x_fusion, alpha=0)
            if xl<3000:
                continue
            _, _, fusion_logits, _, _ = net(x_fusion, alpha=0)

            fusion_logits = fusion_logits.squeeze(0).squeeze(0)

            # save predictions to output map 
            output_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += torch.sigmoid(fusion_logits[mask].cpu()).to(torch.float16) 
            # output_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += fusion_logits[mask].cpu().to(torch.float16) 
            output_map_count[xl:xl+ips, yl:yl+ips][mask.cpu()] += 1
    

    # plot_2dmatrix(torch.sigmoid(output_map.to(torch.float32)))
    # plot_2dmatrix(output_map)
    # plot_2dmatrix(output_map_count)

    # average predictions
    div_mask = output_map_count > 1
    output_map[div_mask] = output_map[div_mask] / output_map_count[div_mask]

    del output_map_count

    # convert to pseudo_probabilities
    # output_map_prob = torch.sigmoid(output_map.to(torch.float32))
    # output_map_prob = output_map_prob.to(torch.float16)

    # output_map_prob[output_map_count==0] = 0
    
    # set the overlap to zero
    output_map[:overlap, :] = 0
    output_map[-overlap:, :] = 0
    output_map[:, :overlap] = 0
    output_map[:, -overlap:] = 0

    # save predictions to file
    region_root = os.path.join(pop_map_root, args.region)
    metadata = dataloader.dataset.metadata()
    metadata.update({"count": 1,
                     "dtype": "float32",
                     "compress": "lzw"
                     })
    
    with rasterio.open(os.path.join(region_root, "buildingsDDA2_4.tif"), 'w', **metadata) as dst:
        dst.write(output_map, 1)
        
    # with rasterio.open(os.path.join(region_root, "buildingsDDA128_4096_nodisc_new_prob_merge.tif"), 'w', **metadata) as dst:
    #     dst.write(output_map_prob, 1)

    return None



if __name__=="__main__":


    # intialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg', '--region', type=str, help='')
    args = parser.parse_args()

    # run main
    main(args)
    print("Done!")