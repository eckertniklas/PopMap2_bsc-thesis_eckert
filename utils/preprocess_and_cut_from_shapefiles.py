import argparse
import fiona
import geopandas as gdp
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
from utils import plot_2dmatrix
from shapely.affinity import translate

def process(hd_regions_path, wp_regions_path,
            census_data_path, output_path, dataset_name, target_col,
            xoff=None, yoff=None):

    
    hd_regions = gdp.read_file(hd_regions_path)
    hd_regions["idx"] = np.nan
    hd_regions['idx'] = hd_regions['idx'].astype('Int64')

    # hd_regions = gdp.read_file(hd_regions_path)["geometry"]
    
    wp_regions = gdp.read_file(wp_regions_path)[["adm_id", "geometry"]]
    
    # all_census = read_multiple_targets_from_csv(census_data_path)
    all_census = pd.read_csv(census_data_path)[["ISO","GID", target_col]]
    all_census = all_census.rename(columns={'ISO': 'ISO', 'GID': 'adm_id', target_col: "pop_count"})

    wp_joined = pd.concat([wp_regions, all_census], axis=1, join="inner")
    
    iou = np.zeros((len(hd_regions), len(wp_joined)))
    for i,hd_row in tqdm(hd_regions.iterrows()):
        hd_geometry = hd_row["geometry"]
        hd_regions.loc[i,"idx"] = int(i)
        if xoff is not None or yoff is not None:
                        
            xoff = 0 if xoff is None else xoff
            yoff = 0 if yoff is None else yoff
            hd_geometry = translate(hd_geometry, xoff=-xoff, yoff=-yoff)

        for j, wp_row in wp_joined.iterrows():
            wp_geometry = wp_row["geometry"]
            intersection = hd_geometry.intersection(wp_geometry)
            if not intersection.is_empty:
                union = hd_geometry.union(wp_geometry)
                iou[i,j] = intersection.area / union.area


    print("Mean ioU matching score", iou.max(1).mean())
    iou[iou<0.5] = 0.

    valid_matches = iou.sum(1)>=0.5
    print("Number of valid matches", sum(valid_matches))
    
    iou_argmax = iou.argmax(1)

    hd_regions["pop_count"] = hd_regions.apply(lambda row: wp_joined["pop_count"][iou_argmax[row["idx"]]], axis=1)
    
    hd_regions = hd_regions[valid_matches] 
    #(minx, miny, maxx, maxy) as a list
    hd_regions = pd.concat([hd_regions, hd_regions["geometry"].bounds], axis=1)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hd_regions_path", type=str, help="Shapefile with humdata.org administrative boundaries information")
    parser.add_argument("wp_regions_path", type=str, help="Shapefile with WorldPop administrative boundaries information")
    parser.add_argument("census_data_path", type=str, help="CSV file containing the WorldPop ids of the regions and the population counts")
    parser.add_argument("output_path", type=str, help="Output path")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    parser.add_argument("target_col", type=str, help="Target column")
    args = parser.parse_args()

    process(args.hd_regions_path, args.wp_regions_path,
                    args.census_data_path, args.output_path, args.dataset_name,
                    args.target_col,
                    # yoff=-0.)
                    yoff=-0.0026)


if __name__ == "__main__":
    main()
    print("Done!")


