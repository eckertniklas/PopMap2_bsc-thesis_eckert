
from os.path import isfile, join, exists 
import os
from os import makedirs, walk, remove
import glob

import pandas as pd
import geopandas as gpd
import numpy as np
from osgeo import gdal
from rasterio.warp import transform_geom
from rasterio.features import is_valid_geom
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.windows import Window
from rasterio import CRS
from rasterio.transform import from_origin
import rasterio
from rasterio import features
from rasterio.features import rasterize

from tqdm import tqdm, tqdm_pandas
import scipy
from PIL import Image
from PIL.Image import Resampling as PILRes
from shapely.geometry import Point
from affine import Affine

import torch

tqdm.pandas()
from utils import plot_2dmatrix


# global
cx = 100
cy = 100

# pseudoscaling
ps = 20

import math


# def calculate_adjacency(df):
#     """
#     Calculate adjacency matrix for a GeoDataFrame.
    
#     Parameters:
#     df (GeoDataFrame): GeoDataFrame containing the polygons.
    
#     Returns:
#     DataFrame: Pandas DataFrame representing the adjacency matrix.
#     """
#     n = len(df)
#     # print("Number of regions:", n)
#     adjacency_matrix = pd.DataFrame(np.zeros((n, n), dtype=bool), index=df.index, columns=df.index)

#     # calculate bounds
#     bounds = df.bounds

#     print("Calculating adjacency matrix")
#     for i, row_i in tqdm(df.iterrows(), total=len(df), leave=True): 
#         for j in range(i+1, n): 
#             # check if the bounds intersect
#             if not (bounds.loc[i, "minx"] <= bounds.loc[j, "maxx"] and bounds.loc[i, "maxx"] >= bounds.loc[j, "minx"] and \
#                     bounds.loc[i, "miny"] <= bounds.loc[j, "maxy"] and bounds.loc[i, "maxy"] >= bounds.loc[j, "miny"]):
#                 continue
            
#             row_j = df.iloc[j]
#             is_adjacent = row_i['geometry'].touches(row_j['geometry'])
#             adjacency_matrix.loc[i, j] = is_adjacent
#             adjacency_matrix.loc[j, i] = is_adjacent # the adjacency matrix is symmetric
    
#     return adjacency_matrix


def calculate_adjacency(df):
    """
    Optimized version to calculate adjacency matrix for a GeoDataFrame.

    Parameters:
    df (GeoDataFrame): GeoDataFrame containing the polygons.

    Returns:
    DataFrame: Pandas DataFrame representing the adjacency matrix.
    """

    n = len(df)
    bounds = df.bounds
    adjacency_matrix = pd.DataFrame(np.zeros((n, n), dtype=bool), index=df.index, columns=df.index)
    
    for i, row_i in tqdm(df.iterrows(), total=len(df)):
        row_i_bounds = bounds.loc[i]
        mask = (bounds['minx'] <= row_i_bounds['maxx']) & (bounds['maxx'] >= row_i_bounds['minx']) & \
               (bounds['miny'] <= row_i_bounds['maxy']) & (bounds['maxy'] >= row_i_bounds['miny'])
        potential_adjacent = df.loc[mask.index[mask]]
        
        for j, row_j in potential_adjacent.iterrows():
            if i != j:
                is_adjacent = row_i['geometry'].touches(row_j['geometry'])
                adjacency_matrix.loc[i, j] = is_adjacent
                adjacency_matrix.loc[j, i] = is_adjacent # the adjacency matrix is symmetric
    
    return adjacency_matrix


def update_adjacency_matrix(matrix, min_pair):
    """
    Updates the adjacency matrix after merging two polygons.
    
    Parameters:
    matrix (DataFrame): The adjacency matrix.
    min_pair (tuple): The indices of the polygons that were merged. 
    
    Returns:
    DataFrame: The updated adjacency matrix.
    """

    new_idx = matrix.index.max() + 1
    
    # copy the dataframe
    new_matrix = matrix.copy()
    new_matrix = new_matrix.astype(bool)

    i, j = min_pair
    
    # New adjacency list for the merged polygon is the union of the adjacency lists of the original polygons
    new_adjacency = new_matrix.loc[i] | new_matrix.loc[j]
    
    # Remove self-adjacency for the merged polygon
    new_adjacency.loc[[i, j]] = False
    
    # Add new row and column for the merged polygon
    new_matrix[new_idx] = new_adjacency

    # Append new row and update the DataFrame in place
    new_adjacency.name = new_idx
    new_matrix = pd.concat([new_matrix, pd.DataFrame(new_adjacency.astype(bool)).T])
    new_matrix.loc[new_idx, new_idx] = False
    # new_matrix = new_matrix.append(new_adjacency, ignore_index=False)
    
    # Remove rows and columns for the original polygons
    new_matrix = new_matrix.drop([i, j], axis=0)
    new_matrix = new_matrix.drop([i, j], axis=1)

    
    return new_matrix


# def find_min_adjacent_pair(df, adjacency_matrix, areas):
#     """
#     Find the pair of adjacent polygons that result in the minimum area when merged.
    
#     Parameters:
#     df (GeoDataFrame): GeoDataFrame containing the polygons.
#     adjacency_matrix (DataFrame): Adjacency matrix.
#     areas (Series): Series containing the area of each polygon.

#     Returns:
#     Tuple: Indices of the pair to be merged.
#     """
#     min_area = float('inf')
#     min_pair = None
#     # print("Finding min adjacent pair...")
#     for i, row in tqdm(adjacency_matrix.iterrows(), leave=False, disable=True):
#         for j, adjacent in row.items():
#             if adjacent:
#                 merged_area = areas[i] + areas[j]
                
#                 if merged_area < min_area:
#                     min_area = merged_area
#                     min_pair = (i, j)
    
#     return min_pair, merged_area


def find_min_adjacent_pair(df, adjacency_matrix, areas):
    """
    Find the pair of adjacent polygons that result in the minimum area when merged.
    
    Parameters:
    df (GeoDataFrame): GeoDataFrame containing the polygons.
    adjacency_matrix (DataFrame): Adjacency matrix.
    areas (Series): Series containing the area of each polygon.

    Returns:
    Tuple: Indices of the pair to be merged.
    """
    
    # Convert the adjacency matrix and areas to NumPy arrays for efficient computation
    adj_matrix_np = adjacency_matrix.to_numpy()
    areas_np = areas.to_numpy()
    
    # Create a 2D array where each element (i, j) contains the sum of the areas of polygons i and j
    area_sums = areas_np[:, None] + areas_np
    
    # Mask the area sums using the adjacency matrix (we are only interested in adjacent polygons)
    area_sums_masked = np.where(adj_matrix_np, area_sums, np.inf)
    
    # Set the diagonal to infinity to avoid selecting self-pairs
    np.fill_diagonal(area_sums_masked, np.inf)
    
    # Find the minimum value and its index
    min_area = np.min(area_sums_masked)
    min_pair_idx = np.unravel_index(np.argmin(area_sums_masked), area_sums_masked.shape)
    
    # Convert back to the original indices
    min_pair = (adjacency_matrix.index[min_pair_idx[0]], adjacency_matrix.columns[min_pair_idx[1]])
    
    return min_pair, min_area


def simplify_shapefile(regionsdf, n=200):
    """
    Simplify the shapefile such that only n samples remain
    """

    # copy the dataframe
    regionsdf_cp = regionsdf.copy()
    crs = regionsdf_cp.crs

    # check if crs is available
    if crs is None:
        crs = CRS.from_epsg(2056)

    # get area of each region
    areas = regionsdf_cp["geometry"].area
    regionsdf_cp["area"] = areas

    # get adjacent regions
    adjacency_matrix = calculate_adjacency(regionsdf_cp)

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(regionsdf_cp) - n)

    # Loop to merge polygons until only 200 are left
    with tqdm(total=len(regionsdf_cp) - n) as pbar:
        # Loop to merge polygons until only n are left
        while len(regionsdf_cp) > n:
            
            assert all(regionsdf_cp.index==adjacency_matrix.index) , "Indices of the adjacency matrix do not match the indices of the GeoDataFrame"
            (i, j), merged_area = find_min_adjacent_pair(regionsdf_cp, adjacency_matrix, areas)
            
            # Merge polygons i and j
            new_polygon = regionsdf_cp.loc[i, 'geometry'].union(regionsdf_cp.loc[j, 'geometry'])
            
            # Add new merged polygon to DataFrame and remove old ones
            new_row = regionsdf_cp.loc[i].copy()
            new_row.name = regionsdf_cp.index.max() + 1
            regionsdf_cp = regionsdf_cp.drop([i, j], axis=0)

            new_row['geometry'] = new_polygon
            new_row['area'] = merged_area
            regionsdf_cp = pd.concat([regionsdf_cp, new_row.to_frame().T])

            # Update adjacent matrix [new index must be +3 because of the appended row and the dropped rows]
            adjacency_matrix = update_adjacency_matrix(adjacency_matrix, min_pair=(i, j))

            # Update areas, kick out indices (i,j) by overwriting the areas vector with the new areas
            areas = regionsdf_cp["area"]

            # Print progress
            pbar.set_postfix({"Min_Pair": (i, j), "Regions_Remaining": len(regionsdf_cp)})
            pbar.update(1)

    # reset the index of the dataframe
    regionsdf_cp.reset_index(drop=True, inplace=True)

    plot = False
    if plot:
        import matplotlib.pyplot as plt

        # Save the plot
        fig, ax = plt.subplots(1, figsize=(10, 10))
        regionsdf_cp.boundary.plot(ax=ax, color='black')
        regionsdf_cp.plot(ax=ax, cmap='Pastel1')
        plt.savefig(os.path.join("plot_outputs", "last_gdfplot.png"))

    # set crs
    regionsdf_cp = regionsdf_cp.set_crs(crs)

    return regionsdf_cp


def compute_pixel_area(transform, crs_in_degrees, mean_latitude=None):
    if not crs_in_degrees:
        return abs(transform[0] * transform[4])
    else:
        # Approximate meters per degree at the given latitude
        m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * mean_latitude) + 1.175 * math.cos(4 * mean_latitude)
        m_per_deg_lon = 111412.84 * math.cos(mean_latitude) - 93.5 * math.cos(3 * mean_latitude)

        pixel_height_in_m = m_per_deg_lat * abs(transform[4])
        pixel_width_in_m = m_per_deg_lon * abs(transform[0])
        
        return pixel_height_in_m * pixel_width_in_m
    

def progress_cb(complete, message, cb_data):
    '''Emit progress report in numbers for 10% intervals and dots for 3% in the classic GDAL style'''
    if int(complete*100) % 10 == 0:
        print(f'{complete*100:.0f}', end='', flush=True)
    elif int(complete*100) % 3 == 0:
        print(f'.', end='', flush=True)
        

# Create a function to check if a point is within the borders of Switzerland using the shapely polygon
def point_within_borders(x, y, polygon):
    point = Point(x, y)
    return polygon.contains(point)


def rasterize_csv(csv_filename, source_popNN_file, source_popBi_file, template_file, source_reprojected_file, output_dir,
                  gpu_mode=True, force=False):
    # definition
    resample_alg = gdal.GRA_Cubic

    # read
    # if not isfile(source_popBi_file):
    # if True:
    if not isfile(os.path.join(output_dir, "boundaries_finezurich2.tif")) or force:
        # df = pd.read_csv(csv_filename)[["E_KOORD", "N_KOORD", "B17BTOT"]]
        df = pd.read_csv(csv_filename, sep=";")[["E_KOORD", "N_KOORD", "B20BTOT"]] 

        # load switzerland shapefile
        gdf = gpd.read_file('/scratch2/metzgern/HAC/data/PopMapData/raw/geoboundaries3d/swissboundaries3d_2021-01_2056_5728.shp/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET.shp')
        switzerland_gdf = gdf[gdf["NAME"] == "Schweiz"]

        # get the min and max coordinates
        E_min = df["E_KOORD"].min()
        N_min = df["N_KOORD"].max()-1
        w = ( df["E_KOORD"].max() - df["E_KOORD"].min() )//cx + 1
        h = ( df["N_KOORD"].max() - df["N_KOORD"].min() )//cy + 1

        # create image coordinates
        df["E_IMG"] = (df["E_KOORD"] - E_min) // cx
        df["N_IMG"] = -(df["N_KOORD"] - N_min) // cy

        # initialize the rasters
        pop_raster = np.zeros((h,w))
        # pop_raster[df["N_IMG"].tolist(), df["E_IMG"].to_list()] = df["B17BTOT"]
        pop_raster[df["N_IMG"].tolist(), df["E_IMG"].to_list()] = df["B20BTOT"]

        # Rasterize the shapefile to create a binary mask
        transform = Affine.translation(E_min, N_min) * Affine.scale(cx, -cy)
        binary_mask = features.rasterize([(geom, 1) for geom in switzerland_gdf.geometry], out_shape=(h, w), transform=transform)
        
        # Create a dataframe with the image coordinates and population counts by checking which pixels are within the borders of Switzerland
        E_IMG, N_IMG = np.meshgrid(np.arange(w), np.arange(h))
        E_KOORD = E_min + E_IMG * cx
        N_KOORD = N_min - N_IMG * cy
        B20BTOT = df.set_index(["E_IMG", "N_IMG"])["B20BTOT"].reindex(pd.MultiIndex.from_arrays([E_IMG.ravel(), N_IMG.ravel()])).fillna(0).values
        final_df = pd.DataFrame({"E_KOORD": E_KOORD.ravel(), "N_KOORD": N_KOORD.ravel(), "B20BTOT": B20BTOT, "E_IMG": E_IMG.ravel(), "N_IMG": N_IMG.ravel()})
        final_df = final_df[binary_mask.ravel() == 1]

        # create enumeration raster
        enumeration_raster = np.zeros((h,w))
        enumeration_raster[final_df["N_IMG"].tolist(), final_df["E_IMG"].to_list()] = np.arange(len(final_df))+1

        # create metadata for the raster
        meta = {"driver": "GTiff", "count": 1, "dtype": "float32", "width":w, "height":h, "crs": CRS.from_epsg(2056),
                "transform": from_origin(E_min, N_min, cx, cy), "compress": "lzw"}
    
        # write the rasters to temporary files
        with rasterio.open("tmp.tif", 'w', **meta) as dst1:
            dst1.write(pop_raster,1)
        with rasterio.open("tmp2.tif", 'w', **meta) as dst1:
            dst1.write(enumeration_raster,1)

        # reproject the rasters to EPSG:3035
        with rasterio.open("tmp.tif", "r") as src:
            src_crs = src.crs
            transform, width, height = calculate_default_transform(src_crs, "EPSG:3035", src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()

            xres, _, ulx, _, yres, uly = transform[0], transform[1], transform[2], transform[3], transform[4], transform[5] 

        # resampling the census grid via nearest neighbour to a higher resolution grid
        with rasterio.open("tmp.tif", "r") as src:
            src_data = src.read(1)

            # adjust and match the total count
            adjusted_data = src_data #* 2913457700.0 / 8742213.0 * adjustment_factor

            kwargs.update({
                'crs': 3035,
                'transform': rasterio.Affine(xres//ps, 0.0, ulx, 0.0, yres//ps, uly),
                # 'transform': transform,
                'width': width*ps,
                'height': height*ps,
                "compress": "lzw"})
                    
            # Update the data with the adjusted values
            with rasterio.open(source_popNN_file, 'w', **kwargs) as dst:
                reproject(
                    source=adjusted_data,  # Use the adjusted data
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:3035",
                    resampling=Resampling.nearest)
                

        # reproject the enumeration raster
        with rasterio.open("tmp2.tif", "r") as src2:
            src2_data = src2.read(1)

            with rasterio.open("raw_enumeration.tif", "w", **kwargs) as dst:
                reproject(
                    source=src2_data,  # Use the adjusted data
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src2.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:3035",
                    resampling=Resampling.nearest)
                
        remove("tmp.tif")
        remove("tmp2.tif")
    
        # adjust the total counts:
        with rasterio.open(source_popNN_file, "r") as src2:
            data = src2.read(1)
            src2meta = src2.meta.copy()

        data = data * src_data.sum() / data.sum()

        src2meta.update({"compress": "lzw"})
        with rasterio.open(source_popNN_file, "w", **src2meta) as dst:
            dst.write(data, 1)


        # Read the template raster
        with rasterio.open(template_file, "r") as template:
            # Get the target transform, CRS, width, and height
            target_transform = template.transform
            target_crs = template.crs
            target_width = template.width
            target_height = template.height
            target_metadata = template.meta.copy()
        

        # Open the source raster to be reprojected and reproject it into the sentinel-1/-2 raster
        with rasterio.open(source_popNN_file, "r") as src:
            # Reproject the source raster to match the template raster
            reprojected_data = np.empty((target_height, target_width), dtype=src.meta["dtype"])
            reproject(
                source=rasterio.band(src, 1),
                destination=reprojected_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
            
            reprojected_data = reprojected_data * src_data.sum() / reprojected_data.sum()


        # Write the reprojected data to a new file (or update the source file)
        target_metadata.update({"compress": "lzw", "count": 1, "dtype": "float32"})
        with rasterio.open(source_reprojected_file, "w", **target_metadata) as dst:
            dst.write(reprojected_data, 1)

        # reproject the enumeration regions as well
        with rasterio.open("raw_enumeration.tif", "r") as src:
            reprojected_enumeration_data = np.empty((target_height, target_width), dtype=src.meta["dtype"])
            reproject(
                source=rasterio.band(src, 1),
                destination=reprojected_enumeration_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )

        # Write the reprojected data to a new file (or update the source file)
        target_metadata.update({"compress": "lzw", "count": 1, "dtype": "int32"})
        with rasterio.open(os.path.join(output_dir, "boundaries_fine.tif"), "w", **target_metadata) as dst:
            dst.write(reprojected_enumeration_data, 1)

        remove("raw_enumeration.tif")
        print("Done reprojecting")

        # TODO: create small subdataset for the greater region of zurich
        xmin, xmax = 2000, 20000
        ymin, ymax = 29800, 30100
        reprojected_data_zurich = np.zeros_like(reprojected_data)
        reprojected_enumeration_data_zurich = np.zeros_like(reprojected_enumeration_data)
        reprojected_data_zurich[xmin:xmax, ymin:ymax] = reprojected_data[xmin:xmax, ymin:ymax]
        reprojected_enumeration_data_zurich[xmin:xmax, ymin:ymax] = reprojected_enumeration_data[xmin:xmax, ymin:ymax]

        # save to file
        finezurich_metadata = target_metadata.copy()
        finezurich_metadata.update({"compress": "lzw", "count": 1, "dtype": "float32", "transform": target_transform, "height": target_height, "width": target_width})
        with rasterio.open(os.path.join(output_dir, "boundaries_finezurich.tif"), "w", **finezurich_metadata) as dst:
            dst.write(reprojected_enumeration_data_zurich, 1)

        # create small subdataset for the greater region of zurich
        xmin, xmax = 2000, 20000
        ymin, ymax = 28000, 31000
        reprojected_data_zurich2 = np.zeros_like(reprojected_data)
        reprojected_enumeration_data_zurich2 = np.zeros_like(reprojected_enumeration_data)
        reprojected_data_zurich2[xmin:xmax, ymin:ymax] = reprojected_data[xmin:xmax, ymin:ymax]
        reprojected_enumeration_data_zurich2[xmin:xmax, ymin:ymax] = reprojected_enumeration_data[xmin:xmax, ymin:ymax]

        # save to file
        finezurich_metadata2 = target_metadata.copy()
        finezurich_metadata2.update({"compress": "lzw", "count": 1, "dtype": "float32", "transform": target_transform, "height": target_height, "width": target_width})
        with rasterio.open(os.path.join(output_dir, "boundaries_finezurich2.tif"), "w", **finezurich_metadata2) as dst:
            dst.write(reprojected_enumeration_data_zurich2, 1)

        # create fine census
        fine_dict = {
            # "finezurich": (reprojected_data_zurich, reprojected_enumeration_data_zurich),
            # "finezurich2": (reprojected_data_zurich2, reprojected_enumeration_data_zurich2),
            # "fine": (reprojected_data, reprojected_enumeration_data),
        }

        for name, (popdensity, boundaries) in fine_dict.items():
            
            if not os.path.exists(os.path.join(output_dir, "census_" + name + ".csv")) or force:

                if gpu_mode:
                    popdensity = torch.from_numpy(popdensity)
                    boundaries = torch.from_numpy(boundaries).cuda()
                else:
                    popdensity = torch.from_numpy(popdensity)
                    boundaries = torch.from_numpy(boundaries)

                # get unique indices of boundaries
                unique_indices = torch.unique(boundaries)
                
                # create a dataframe to store the census data
                n_rows = len(unique_indices)
                all_census = pd.DataFrame(index=range(n_rows))

                all_census["idx"] = 0
                all_census["POP20"] = 0
                all_census["bbox"] = ""
                all_census["count"] = 0

                # get the bounding box and count counts of each region
                for rowi, census_idx in tqdm(enumerate(unique_indices), total=len(unique_indices)):
                    if census_idx==0:
                        xmin, ymax, ymin, ymax = 0, 0, 0, 0
                        popcount = -1
                        count = 1e-12
                    else:
                        mask = boundaries==census_idx
                        vertical_indices = torch.where(torch.any(mask, dim=1))[0]
                        horizontal_indices = torch.where(torch.any(mask, dim=0))[0]
                        xmin, xmax = vertical_indices[[0,-1]].cpu()
                        ymin, ymax = horizontal_indices[[0,-1]].cpu()
                        xmax, ymax = xmax+1, ymax+1
                        xmin, xmax, ymin, ymax = xmin.item(), xmax.item(), ymin.item(), ymax.item()

                        mask_slice = mask[xmin:xmax, ymin:ymax]
                        count = mask_slice.sum().item()

                        # check if the region is valid
                        if count==0:
                            xmin, ymax, ymin, ymax = 0, 0, 0, 0
                            popcount = 0
                        else:
                            if gpu_mode:
                                popcount = popdensity[xmin:xmax, ymin:ymax].cuda()[mask_slice].sum().item()
                            else:
                                popcount = popdensity[xmin:xmax, ymin:ymax][mask_slice].sum().item()

                    all_census.loc[rowi, "idx"] = census_idx.item()
                    all_census.loc[rowi, "POP20"] = popcount
                    all_census.at[rowi, "bbox"] = [xmin, xmax, ymin, ymax]
                    all_census.loc[rowi, "count"] = count

                # weed out the regions that are invalid
                all_census = all_census[all_census["POP20"]!=-1]

                all_census[["idx", "POP20", "bbox", "count"]].to_csv(os.path.join(output_dir, "census_" + name + ".csv"))

        del fine_dict
    
    else:

        with rasterio.open(template_file, "r") as template:
            # Get the target transform, CRS, width, and height
            target_transform = template.transform
            target_crs = template.crs
            target_width = template.width
            target_height = template.height
            target_metadata = template.meta.copy()

        with rasterio.open(source_reprojected_file, "r") as dst:
            reprojected_data = dst.read(1)
        
        
    # load shapefile
    boundaries = { 
        "boundaries_coarse4synt100": "/scratch2/metzgern/HAC/data/PopMapData/raw/geoboundaries3d/swissboundaries3d_2021-01_2056_5728.shp/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp",
        "boundaries_coarse4synt200": "/scratch2/metzgern/HAC/data/PopMapData/raw/geoboundaries3d/swissboundaries3d_2021-01_2056_5728.shp/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp",
        "boundaries_coarse4synt400": "/scratch2/metzgern/HAC/data/PopMapData/raw/geoboundaries3d/swissboundaries3d_2021-01_2056_5728.shp/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp",
        "boundaries_coarse4": "/scratch2/metzgern/HAC/data/PopMapData/raw/geoboundaries3d/swissboundaries3d_2021-01_2056_5728.shp/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.shp",
        # "boundaries_coarse3": "/scratch2/metzgern/HAC/data/PopMapData/raw/geoboundaries3d/swissboundaries3d_2021-01_2056_5728.shp/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_HOHEITSGRENZE.shp",
        "boundaries_coarse2": "/scratch2/metzgern/HAC/data/PopMapData/raw/geoboundaries3d/swissboundaries3d_2021-01_2056_5728.shp/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_BEZIRKSGEBIET.shp",
        "boundaries_coarse1": "/scratch2/metzgern/HAC/data/PopMapData/raw/geoboundaries3d/swissboundaries3d_2021-01_2056_5728.shp/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp",
    }
    census_names = {
        "boundaries_coarse4": "census_coarse4",
        "boundaries_coarse4synt100": "census_coarse4synt100",
        "boundaries_coarse4synt200": "census_coarse4synt200",
        "boundaries_coarse4synt400": "census_coarse4synt400",
        # "boundaries_fine3": "census_coarse3",
        "boundaries_coarse2": "census_coarse2",
        "boundaries_coarse1": "census_coarse1",
    }

    for bname,path in tqdm(boundaries.items()):

        # read the boundaries
        # regionsdf = gpd.read_file(boundaries)[["geometry"]]
        regionsdf = gpd.read_file(path)

        if bname=="boundaries_coarse4synt200":
            # simplify the shapefiles such that only 200 samples remain
            regionsdf = simplify_shapefile(regionsdf, n=200)

        elif bname=="boundaries_coarse4synt400":
            # simplify the shapefiles such that only 400 samples remain
            regionsdf = simplify_shapefile(regionsdf, n=400)
        
        elif bname=="boundaries_coarse4synt100":
            # simplify the shapefiles such that only 100 samples remain
            regionsdf = simplify_shapefile(regionsdf, n=100)


        # add index, that starts at 1
        regionsdf["idx"] = np.arange(len(regionsdf))+1
        # regionsdf["idx"] = np.arange(len(regionsdf))

        # reproject
        regionsdf = regionsdf.to_crs(target_crs)
        shapes = []
        for geom, value in tqdm(zip(regionsdf.geometry, regionsdf["idx"]), total=len(regionsdf), disable=True):
            shapes.append((geom, value))
        raster_data = rasterize(shapes, out_shape=(target_height, target_width), transform=target_transform)

        # Save the rasterized data to a new GeoTIFF file
        output_path = os.path.join(output_dir, bname+".tif")

        target_metadata.update({"driver": "GTiff", "height": target_height, "width": target_width,
                                "transform": target_transform, "crs": target_crs, "count": 1, "compress": "lzw", "dtype": np.uint16 })
        with rasterio.open(output_path, 'w', **target_metadata) as dst:
            dst.write(raster_data, 1)

        print("Done rasterizing", bname)

        if gpu_mode: 
            burned = torch.from_numpy(raster_data.astype(np.int16)).cuda()
            popdensity = torch.from_numpy(reprojected_data.astype(np.float32))
        else:
            burned = torch.from_numpy(raster_data.astype(np.int16))
            popdensity = torch.from_numpy(reprojected_data.astype(np.float32))

        regionsdf["POP20"] = 0
        regionsdf["bbox"] = ""
        regionsdf["count"] = 0

        # get the bounding box and count of each region to enrich the data, also add the count to the dataframe  
        for rowi, row in enumerate(tqdm(regionsdf.itertuples(), total=len(regionsdf))): 
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

            # check if the region is valid
            if count==0:
                popcount = 0
                xmin, ymax, ymin, ymax = 0, 0, 0, 0
            else:
                popcount = popdensity[xmin:xmax, ymin:ymax].cuda()[mask[xmin:xmax, ymin:ymax]].sum().item()
            
            # check if the region is in switzerland
            if pd.isnull(row.KANTONSNUM):
                # continue
                regionsdf.at[rowi, "POP20"] = -1

            regionsdf.at[rowi, "POP20"] = popcount
            regionsdf.at[rowi, "bbox"] = [xmin, xmax, ymin, ymax]
            regionsdf.loc[rowi, "count"] = count

        # weed out the regions that are not in switzerland
        regionsdf = regionsdf[regionsdf["POP20"]!=-1]

        # regionsdf[["idx", "POP20", "bbox", "count", "EINWOHNERZ"]].to_csv(os.path.join(output_dir, name+".csv"))
        regionsdf[["idx", "POP20", "bbox", "count"]].to_csv(os.path.join(output_dir, census_names[bname]+".csv"))
        
        print("Done creating census", bname)
    print("Done")


def process():
    # source_folder = "/scratch2/metzgern/HAC/data/BFS_CH/2017"
    source_folder = "/scratch2/metzgern/HAC/data/BFS_CH/2020"
    # source_filename = "STATPOP2017.csv"
    source_filename = "STATPOP2020.csv"
    source_meta_poprasterNN = "PopRasterNN.tif"
    source_meta_poprasterBi = "PopRasterBi.tif"
    source_reprojected = "reprojected_popNN.tif"

    template_file = "/scratch2/metzgern/HAC/data/PopMapData/merged/EE/che/S1summer/che_S1summer.tif"

    # output_dir
    output_dir = '/scratch/metzgern/HAC/data/PopMapData/processed/che'

    source_file = join(source_folder, source_filename)
    source_popNN_file = join(source_folder, source_meta_poprasterNN)
    source_popBi_file = join(source_folder, source_meta_poprasterBi)
    source_reprojected_file = join(source_folder, source_reprojected)
    rasterize_csv(source_file, source_popNN_file, source_popBi_file, template_file, source_reprojected_file, output_dir, force=False)
    
    return


if __name__=="__main__":
    process()
    print("Done")
