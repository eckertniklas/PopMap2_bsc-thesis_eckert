import argparse
import fiona
import geopandas as gdp
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
from plot import plot_2dmatrix
from shapely.affinity import translate
import rasterio
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.warp import transform_bounds

# Affine transformation
from rasterio import Affine

import os
import torch
import matplotlib.pyplot as plt


def process(hd_regions_path,
            census_data_path, output_path, dataset_name, target_col,
            template_file,
            xoff=None, yoff=None, gpu_mode=True):

    
    hd_regions = gdp.read_file(hd_regions_path)

    plot = False
    if plot:
        fig, ax = plt.subplots(1, 1)
        hd_regions.plot(ax=ax)
        plt.savefig('plot_outputs/shapefile_plot.png', dpi=300)
    
    # all_census = read_multiple_targets_from_csv(census_data_path)
    all_census = pd.read_csv(census_data_path, encoding='ISO-8859-1')[["ADM3_EN","ADM3_PCODE", target_col]]

    # merge the two dataframes via an inner join
    merged_df = pd.merge(all_census, hd_regions, on='ADM3_PCODE')

    # read metadata of the template file
    with rasterio.open(template_file, 'r') as tmp:
        metadata = tmp.meta.copy()
    metadata.update({"count": 1, "dtype": rasterio.int32, "compress": "lzw"})
    
    this_outputfile = os.path.join(output_path, 'boundaries_coarse.tif')
    this_outputfile_densities = os.path.join(output_path, 'densities_coarse.tif')
    this_outputfile_totals = os.path.join(output_path, 'totals_coarse.tif')
    this_censusfile = os.path.join(output_path, 'census_coarse.csv')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    merged_df.reset_index(drop=True, inplace=True)
    merged_df["idx"] = merged_df.index+1

    # rasterize
    metadata.update({"compress": "lzw"})
    with rasterio.open(this_outputfile, 'w+', **metadata) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,i) for i, geom in zip(merged_df["idx"], merged_df.geometry))

        # flattens the shapefile into the raster (burns them in)
        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)

    burned = torch.tensor(burned, dtype=torch.int32)
    if gpu_mode:
        burned = burned.cuda()

    merged_df["bbox"] = ""
    merged_df["count"] = 0

    # get the bounding box and count of each region to enrich the data, also add the count to the dataframe  
    for rowi, row in enumerate(tqdm(merged_df.itertuples(), total=len(merged_df))):
        i = row.idx
        mask = burned==i
        count = mask.sum().cpu().item()
        if count==0:
            xmin, ymax, ymin, ymax = 0, 0, 0, 0
        else:
            vertical_indices = torch.where(torch.any(mask, dim=1))[0]
            horizontal_indices = torch.where(torch.any(mask, dim=0))[0]
            xmin, xmax = vertical_indices[[0,-1]].cpu()
            ymin, ymax = horizontal_indices[[0,-1]].cpu()
            xmax, ymax = xmax+1, ymax+1
            xmin, xmax, ymin, ymax = xmin.item(), xmax.item(), ymin.item(), ymax.item()

        merged_df.at[rowi, "bbox"] = [xmin, xmax, ymin, ymax]
        merged_df.loc[rowi, "count"] = count

    merged_df["POP20"] = merged_df["TOTAL22"]
    merged_df[["idx", "POP20", "bbox", "count", "ADM3_EN_x", "ADM3_PCODE"]].to_csv(this_censusfile)

    # create map of densities
    densities = torch.zeros_like(burned, dtype=torch.float32)
    totals = torch.zeros_like(burned, dtype=torch.float32)
    for row in merged_df.itertuples():
        densities[burned==row.idx] = row.POP20/row.count
        totals[burned==row.idx] = row.POP20

    if burned.is_cuda:
        burned = burned.cpu()
        densities = densities.cpu()
        totals = totals.cpu()

    #save densities
    metadatad = metadata.copy()
    metadatad.update({"dtype": rasterio.float32, "compress": "lzw"})
    with rasterio.open(this_outputfile_densities, 'w+', **metadatad) as out:
        out.write_band(1, densities.numpy())

    #save totals
    metadatad = metadata.copy()
    metadatad.update({"dtype": rasterio.float32, "compress": "lzw"})
    with rasterio.open(this_outputfile_totals, 'w+', **metadatad) as out:
        out.write_band(1, totals.numpy())

    print("Done with Rwanda 2022")

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hd_regions_path", type=str, help="Shapefile with humdata.org administrative boundaries information https://data.humdata.org/dataset/cod-ab-rwa")
    parser.add_argument("--census_data_path", type=str, help="data of the 5th rwanda census counts 2022")
    parser.add_argument("--output_path", type=str, help="Output path")
    parser.add_argument("--dataset_name", type=str, help="Dataset name")
    parser.add_argument("--target_col", type=str, help="Target column")
    parser.add_argument("--template_file", type=str, help="template Sentinel-2/1 file that shows the resolution of the output")
    #/scratch2/metzgern/HAC/data/pop_growth_dataset/ancillary_data
    args = parser.parse_args()

    process(args.hd_regions_path,
                    args.census_data_path, args.output_path,
                    args.dataset_name,
                    args.target_col,
                    args.template_file)


if __name__ == "__main__":
    main()
    print("Done!")


