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


def process(target_regions_path, wp_regions_path,
            census_data_path, output_path, dataset_name, target_col,
            template_file,
            xoff=None, yoff=None, gpu_mode=True):

    
    target_regions = gdp.read_file(target_regions_path)
    target_regions["idx"] = np.nan
    target_regions['idx'] = target_regions['idx'].astype('Int64')

    # hd_regions = gdp.read_file(target_regions_path)["geometry"]
    
    wp_regions = gdp.read_file(wp_regions_path)[["adm_id", "geometry"]]

    plot = False
    if plot:
        fig, ax = plt.subplots(1, 1)
        wp_regions.plot(ax=ax)
        plt.savefig('plot_outputs/shapefile_plot.png', dpi=300)
    
    # all_census = read_multiple_targets_from_csv(census_data_path)
    all_census = pd.read_csv(census_data_path)[["ISO","GID", target_col]]
    all_census = all_census.rename(columns={'ISO': 'ISO', 'GID': 'adm_id', target_col: "pop_count"})

    # wp_joined = pd.concat([wp_regions, all_census], axis=1, join="inner")
    wp_joined2 = wp_regions.merge(all_census, on='adm_id', how='inner')
    
    iou_calc = np.zeros((len(target_regions), len(wp_joined2)))
    for i,hd_row in tqdm(target_regions.iterrows(), total=len(target_regions)):
        hd_geometry = hd_row["geometry"]
        target_regions.loc[i,"idx"] = int(i)
        if xoff is not None or yoff is not None:
                        
            xoff = 0 if xoff is None else xoff
            yoff = 0 if yoff is None else yoff
            hd_geometry = translate(hd_geometry, xoff=-xoff, yoff=-yoff)

        for j, wp_row in wp_joined2.iterrows():
            wp_geometry = wp_row["geometry"]
            intersection = hd_geometry.intersection(wp_geometry)
            if not intersection.is_empty:
                union = hd_geometry.union(wp_geometry)
                iou_calc[i,j] = intersection.area / union.area

    print("Mean IoU matching score", iou_calc.max(1).mean())
    print("Median IoU matching score", np.median(iou_calc.max(1)))

    iou = iou_calc.copy()
    iou_thresh = 0.66
    iou[iou<iou_thresh] = 0.

    valid_matches = iou.sum(1)>=0.5
    print("Number of valid matches", sum(valid_matches))
    
    # hardening the matches
    iou_argmax = iou.argmax(1)


    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_regions_path", type=str, default='/scratch2/metzgern/HAC/data/PopMapData/raw/geoboundaries/tza/geoBoundaries-TZA-ADM3.shp', help="Shapefile with high resolution geoboundaries")
    parser.add_argument("--wp_regions_path", type=str, default='/scratch2/metzgern/HAC/data/PopMapData/raw/WorldPopGIS/Mastergrid/Global_2000_2020/TZA/Subnational/Shapefile/tza_subnational_2000_2020.shp', help="Shapefile with WorldPop administrative boundaries information")
    parser.add_argument("--census_data_path", type=str, help="CSV file containing the WorldPop ids of the regions and the population counts")
    parser.add_argument("--output_path", type=str, default="/scratch2/metzgern/HAC/data/PopMapData/raw/WorldPopGIS/Population/Global_2000_2020/CensusTables/tza_population_2000_2020.csv", help="Output path")
    parser.add_argument("--dataset_name", default="tza", type=str, help="Dataset name")
    parser.add_argument("--target_col", default="P_2020", type=str, help="Target column")
    parser.add_argument("--template_file", type=str, default="/scratch2/metzgern/HAC/data/PopMapData/raw/EE/tza", help="template Sentinel-2/1 file that shows the resolution of the output")
    #/scratch2/metzgern/HAC/data/pop_growth_dataset/ancillary_data
    args = parser.parse_args()

    process(args.target_regions_path, args.wp_regions_path,
                    args.census_data_path, args.output_path,
                    args.dataset_name,
                    args.target_col,
                    args.template_file,
                    # yoff=-0.)
                    yoff=-0.0026)


if __name__ == "__main__":
    main()
    print("Done!")


