
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

import gc

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
    dataSource = driver.Open(sh_census_path, 0)
    layer = dataSource.GetLayer()

    def progress_func(complete, message, user_data):
        user_data.update(1)  # Update the tqdm progress bar by 1 unit
        return 1  # 1 to continue, 0 to stop the algorithm

    
    if not os.path.exists(output_tif_file):
        
        print("Rasterizing to file: ", output_tif_file)
        # Define the output raster
        # rasterDS = gdal.GetDriverByName('GTiff').Create(output_tif_file, width, height, 1, gdal.GDT_Int32, ["COMPRESS=LZW"]) # Use your actual output file path
        rasterDS = gdal.GetDriverByName('GTiff').Create(output_tif_file, width, height, 1, gdal.GDT_Int32, ["COMPRESS=DEFLATE"]) # Use your actual output file path
        rasterDS.SetGeoTransform(geotransform)
        rasterDS.SetProjection(referencesystem)
        band = rasterDS.GetRasterBand(1)
        band.SetNoDataValue(-9999)

        with tqdm(total=100, desc="Rasterizing", unit="%") as pbar:
            # Rasterize with progress callback
            gdal.RasterizeLayer(rasterDS, [1], layer, options=["ATTRIBUTE=idx"], callback=progress_func, callback_data=pbar)
            rasterDS = None

        del rasterDS, band
    else:
        print("Raster file already exists, skipping rasterization")

    # clean up
    gc.collect()
    del wpop_regions, layer, dataSource, driver, vrt, geotransform, referencesystem, metadata, width, height, sh_census_path, files_to_mosaic, all_census

    print("Reading rasterized file: ", output_tif_file)
    # Define chunk size
    chunk_size = 512  # This is an example value, adjust as needed

    # Open the rasterized file
    with rasterio.open(output_tif_file) as src:
        
        # Get width and height of the image
        width, height = src.width, src.height
        
        # Directly initialize torch tensor with the appropriate size and type
        if gpu_mode:
            burned = torch.empty((height, width), dtype=torch.int16, device="cuda")
            # burned = torch.empty((height, width), dtype=torch.int16, device="cuda")
        else:
            burned = np.empty((height, width), dtype=np.int16)
        
        
        # Calculate number of chunks in both dimensions
        chunks_x = int(np.ceil(width / chunk_size))
        chunks_y = int(np.ceil(height / chunk_size))

        # Read in chunks
        for i in tqdm(range(chunks_y)):
            for j in range(chunks_x):
                # Calculate window start and stop coordinates
                start_x = j * chunk_size
                start_y = i * chunk_size
                stop_x = min((j + 1) * chunk_size, width)
                stop_y = min((i + 1) * chunk_size, height)
                
                # Define window
                window = rasterio.windows.Window(col_off=start_x, row_off=start_y,
                                                width=stop_x - start_x, height=stop_y - start_y)
                
                # Read the chunk into the pre-initialized array
                if gpu_mode:
                    chunk_data = src.read(1, window=window).astype(np.int16)
                    burned[start_y:stop_y, start_x:stop_x] = torch.from_numpy(chunk_data).cuda()
                else:
                    chunk_data = src.read(1, window=window).astype(np.int16)
                    burned[start_y:stop_y, start_x:stop_x] = chunk_data
    del src, chunk_data, window, start_x, start_y, stop_x, stop_y, chunks_x, chunks_y, width, height

    # create a dataframe to store the census data
    print("Creating census dataframe to", output_census_file)
    wp_joined2["bbox"] = ""
    wp_joined2["count"] = 0

    # get the bounding box and count of each region to enrich the data, also add the count to the dataframe  
    for rowi, row in enumerate(tqdm(wp_joined2.itertuples(), total=len(wp_joined2))): 
        i = row.idx
        mask = burned==i
        
        if gpu_mode:
            vertical_indices = torch.where(torch.any(mask, dim=1))[0]
            horizontal_indices = torch.where(torch.any(mask, dim=0))[0]
            xmin, xmax = vertical_indices[[0,-1]].cpu()
            ymin, ymax = horizontal_indices[[0,-1]].cpu()
        else:
            vertical_indices = np.where(np.any(mask, axis=1))[0]
            horizontal_indices = np.where(np.any(mask, axis=0))[0]
            xmin, xmax = vertical_indices[[0, -1]]
            ymin, ymax = horizontal_indices[[0, -1]]

        # xmin, xmax = vertical_indices[[0,-1]].cpu()
        # ymin, ymax = horizontal_indices[[0,-1]].cpu()
        xmax, ymax = xmax+1, ymax+1
        xmin, xmax, ymin, ymax = xmin.item(), xmax.item(), ymin.item(), ymax.item()

        # check
        # plot_2dmatrix(burned[xmin:xmax, ymin:ymax].cpu().numpy(), vmin=-1)

        count = mask[xmin:xmax, ymin:ymax].sum().item()
        del mask

        if count==0:
            xmin, ymax, ymin, ymax = 0, 0, 0, 0

        wp_joined2.at[rowi, "bbox"] = [xmin, xmax, ymin, ymax]
        wp_joined2.loc[rowi, "count"] = count


    wp_joined2["POP20"] = wp_joined2["pop_count"]
    wp_joined2[["idx", "POP20", "bbox", "count"]].to_csv(output_census_file)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sh_path", default="/scratch2/metzgern/HAC/data/PopMapData/raw/WorldPopGIS/Mastergrid/Global_2000_2020/AFG/Subnational/Shapefile/afg_subnational_2000_2020.shp", type=str, help="Shapefile with boundaries and census")
    parser.add_argument("--census_path", default="/scratch2/metzgern/HAC/data/PopMapData/raw/WorldPopGIS/Population/Global_2000_2020/CensusTables/afg_population_2000_2020.csv", type=str, help="Shapefile with boundaries and census")
    parser.add_argument("--output_tif_file", default="/scratch/metzgern/HAC/data/PopMapData/processed/afg/boundaries.tif", type=str, help="")
    parser.add_argument("--output_census_file", default="/scratch/metzgern/HAC/data/PopMapData/processed/afg/census.csv", type=str, help="")
    parser.add_argument("--template_file_dir", default="/scratch2/metzgern/HAC/data/PopMapData/raw/EE/afg/S1winter/", type=str, help="")
    args = parser.parse_args()

    process(args.sh_path, args.census_path, args.output_tif_file, args.output_census_file, args.template_file_dir, gpu_mode=False)


if __name__=="__main__":
    main()
    print("Done")
