

import argparse
from osgeo import gdal, osr, ogr
import os
import numpy as np
import geopandas as gdp
import pandas as pd

import matplotlib.pyplot as plt

import rasterio
import torch
from tqdm import tqdm
from plot import plot_2dmatrix

def process(sh_path, census_path, output_tif_file, output_census_file, template_file_dir, gpu_mode=True):



    # process the census file
    wpop_regions = gdp.read_file(sh_path)

    #  read from census_path and merge to df
    target_col = "P_2020"
    all_census = pd.read_csv(census_path)[["ISO","GID", target_col]]
    all_census = all_census.rename(columns={'ISO': 'ISO', 'GID': 'adm_id', target_col: "pop_count"})

    # wp_joined = pd.concat([wp_regions, all_census], axis=1, join="inner")
    wp_joined2 = wpop_regions.merge(all_census, on='adm_id', how='inner')

    plot = False
    if plot:
        fig, ax = plt.subplots(1, 1)
        wp_joined2.plot(ax=ax, column='Id', legend=True)
        plt.savefig('plot_outputs/shapefile_plot.png', dpi=300)

    # reset the index
    wp_joined2 = wp_joined2.reset_index(drop=True)
    wp_joined2["idx"] = np.arange(len(wp_joined2))

    # Save the wp_joined2 as a shapefile
    sh_census_path = sh_path.replace('.shp', '_census.shp')
    wp_joined2.to_file(sh_census_path, driver='ESRI Shapefile')



    # get information about the template files
    files_to_mosaic = [ os.path.join(template_file_dir,file) for file in os.listdir(template_file_dir)]

    vrt = gdal.BuildVRT('output.vrt', files_to_mosaic)
    geotransform = vrt.GetGeoTransform()
    referencesystem = vrt.GetProjection()
    metadata = vrt.GetMetadata() 
    width = vrt.RasterXSize
    height = vrt.RasterYSize

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(sh_census_path, 0) # Use your actual file path
    layer = dataSource.GetLayer()

    # Define the output raster
    # rasterDS = gdal.GetDriverByName('GTiff').Create(output_tif_file, width, height, 1, gdal.GDT_Int32, ["COMPRESS=LZW"]) # Use your actual output file path
    rasterDS = gdal.GetDriverByName('GTiff').Create(output_tif_file, width, height, 1, gdal.GDT_Int32, ["COMPRESS=DEFLATE"]) # Use your actual output file path
    rasterDS.SetGeoTransform(geotransform)
    rasterDS.SetProjection(referencesystem)
    band = rasterDS.GetRasterBand(1)
    band.SetNoDataValue(-9999)

    # Rasterize
    gdal.RasterizeLayer(rasterDS, [1], layer, options=["ATTRIBUTE=idx"])
    rasterDS = None


    # read the rasterized file and load it to cuda
    with rasterio.open(output_tif_file) as src:
        burned = src.read(1).astype(np.int16)
        burned = torch.from_numpy(burned).cuda()

    # create a dataframe to store the census data
    # all_census = pd.DataFrame(columns=["idx", "POP20", "bbox", "count"])

    wp_joined2["bbox"] = ""
    wp_joined2["count"] = 0

    # get the bounding box and count of each region to enrich the data, also add the count to the dataframe  
    for rowi, row in enumerate(tqdm(wp_joined2.itertuples(), total=len(wp_joined2))): 
        i = row.idx
        mask = burned==i
        # count = 1
        # count = mask.sum().cpu().item()
        # count = mask.cpu().sum().item()
        vertical_indices = torch.where(torch.any(mask, dim=1))[0]
        horizontal_indices = torch.where(torch.any(mask, dim=0))[0]
        xmin, xmax = vertical_indices[[0,-1]].cpu()
        ymin, ymax = horizontal_indices[[0,-1]].cpu()
        xmax, ymax = xmax+1, ymax+1
        xmin, xmax, ymin, ymax = xmin.item(), xmax.item(), ymin.item(), ymax.item()

        # check
        # plot_2dmatrix(burned[xmin:xmax, ymin:ymax].cpu().numpy(), vmin=-1)

        count = mask[xmin:xmax, ymin:ymax].sum().item()

        if count==0:
            xmin, ymax, ymin, ymax = 0, 0, 0, 0

        wp_joined2.at[rowi, "bbox"] = [xmin, xmax, ymin, ymax]
        wp_joined2.loc[rowi, "count"] = count

    wp_joined2["POP20"] = wp_joined2["pop_count"]
    wp_joined2[["idx", "POP20", "bbox", "count"]].to_csv(output_census_file)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sh_path", default="/scratch2/metzgern/HAC/data/PopMapData/raw/WorldPopGIS/Mastergrid/Global_2000_2020/UGA/Subnational/Shapefile/uga_subnational_2000_2020.shp", type=str, help="Shapefile with boundaries and census")
    parser.add_argument("--census_path", default="/scratch2/metzgern/HAC/data/PopMapData/raw/WorldPopGIS/Population/Global_2000_2020/CensusTables/uga_population_2000_2020.csv", type=str, help="Shapefile with boundaries and census")
    parser.add_argument("--output_tif_file", default="/scratch/metzgern/HAC/data/PopMapData/processed/uga/boundaries.tif", type=str, help="")
    parser.add_argument("--output_census_file", default="/scratch/metzgern/HAC/data/PopMapData/processed/uga/census.csv", type=str, help="")
    parser.add_argument("--template_file_dir", default="/scratch2/metzgern/HAC/data/PopMapData/raw/EE/uga/S1winter/", type=str, help="")
    args = parser.parse_args()

    process(args.sh_path, args.census_path, args.output_tif_file, args.output_census_file, args.template_file_dir, gpu_mode=True)


if __name__ == "__main__":
    main()
    print("Done!")




