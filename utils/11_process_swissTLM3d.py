
import argparse
import rasterio
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import fiona
from rasterio import features
from rasterio.windows import Window
from itertools import product

from plot import plot_2dmatrix


def process(buildings_shapefile, target_raster, suffix=""):
    """
    Input:
        target_regions_path: path to the shapefile with the building footprints as polygons
        target_raster: path to the raster that shows the resolution of the output
    Output:
        None   
    """

    # building_df = gpd.read_file(buildings_shapefile)
    def load_shapefile_to_geodataframe(buildings_shapefile):
        features = []

        with fiona.open(buildings_shapefile) as reader:
            progressbar = tqdm(total=len(reader))
            
            for element in reader:
                progressbar.update(1)
                if element:
                    features.append(element)
                # if len(features) > 100000:
                #     break
        
        print("Merging features to geodataframe...")
        return gpd.GeoDataFrame.from_features(features)
    
    building_df = load_shapefile_to_geodataframe(buildings_shapefile)
    building_df.crs = "EPSG:2056"
    print("Done loading shapefile:", buildings_shapefile)

    # load the raster
    with rasterio.open(target_raster) as src:
        src_transform = src.transform
        src_crs = src.crs
        src_bounds = src.bounds
        src_height = src.height
        src_width = src.width
        meta = src.meta.copy()

    # Note: Need to calculate the area of the building footprints in the raster coordinate system in a projected (metric unit) coordinate system
    # process the building footprints
    building_df['area'] = building_df['geometry'].area
    building_df = building_df[building_df['area']> 0]

    # calculate centroids
    building_df['centroid'] = building_df['geometry'].centroid

    # convert the building footprints to the raster coordinate system
    building_df = building_df.to_crs(src_crs)
    building_df['centroid'] = building_df['centroid'].to_crs(src_crs)

    # map the centroids and area to the raster
    building_count = np.zeros((src_height, src_width))
    building_area = np.zeros((src_height, src_width))

    # convert centroid to image coordinates
    for centroid, area in tqdm(zip(building_df['centroid'], building_df['area']), total=len(building_df)):
        centroid_x, centroid_y = centroid.x, centroid.y
        lon, lat = ~src_transform * (centroid_x, centroid_y)
        lon, lat = int(lon), int(lat)

        # accumulate the count on the pixel
        building_count[lat, lon] += 1

        # add the area to the pixel (assuming you want to track building area per pixel as well)
        building_area[lat, lon] += area
    
    # Only divide where building_count is not zero. Otherwise, set the value to 0.
    mean_building_area = np.where(building_count > 0.5, building_area / building_count, 0)

    def windowed_write(dst, array):
        # Define a window size
        rows_per_block, cols_per_block = dst.block_shapes[0]
        
        # Calculate number of blocks in both dimensions
        n_rows_blocks = int(np.ceil(array.shape[0] / rows_per_block))
        n_cols_blocks = int(np.ceil(array.shape[1] / cols_per_block))

        # Write data block by block
        for row_block in tqdm(range(n_rows_blocks)):
            for col_block in range(n_cols_blocks):
                # Calculate window position
                start_row = row_block * rows_per_block
                start_col = col_block * cols_per_block
                window = Window(col_off=start_col, row_off=start_row, width=cols_per_block, height=rows_per_block)
                
                # Slice the source data array
                src_data = array[start_row:start_row+rows_per_block, start_col:start_col+cols_per_block]

                # Write the sliced array to the destination
                dst.write(src_data, window=window, indexes=1)


    # save the building count raster
    this_meta = meta.copy()
    this_meta.update({'dtype': 'uint8', 'count': 1, 'compress': 'lzw', 'nodata': None})
    output_path = buildings_shapefile.replace(".shp", "_count_" + suffix + ".tif")
    print("Saving building count raster to:", output_path)
    with rasterio.open(output_path, 'w', **this_meta) as dst:
        windowed_write(dst, building_count)

    #save total building area raster
    this_meta = meta.copy()
    this_meta.update({'dtype': 'float32', 'count': 1, 'compress': 'lzw', 'nodata': -1})
    output_path = buildings_shapefile.replace(".shp", "_total_area_" + suffix + ".tif")
    print("Saving total building area raster to:", output_path)
    with rasterio.open(output_path, 'w', **this_meta) as dst:
        windowed_write(dst, building_area)

    # save the mean building area raster
    this_meta = meta.copy()
    this_meta.update({'dtype': 'float32', 'count': 1, 'compress': 'lzw', 'nodata': -1})
    output_path = buildings_shapefile.replace(".shp", "_mean_area_" + suffix + ".tif")
    print("Saving building area raster to:", output_path)
    with rasterio.open(output_path, 'w', **this_meta) as dst:
        windowed_write(dst, mean_building_area)

    # clean up
    del building_count
    del building_area

    # rasterize
    this_meta = meta.copy()
    this_meta.update({'dtype': 'uint8', 'count': 1, 'compress': 'lzw', 'nodata': None})
    output_path = buildings_shapefile.replace(".shp", "_segmentation_" + suffix + ".tif")
    print("Saving building segmentation raster to:", output_path)
    with rasterio.open(output_path, 'w+', **this_meta) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = building_df.geometry

        # flattens the shapefile into the raster (burns them in)
        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform, all_touched=True)
        out.write_band(1, burned)

    print("Done")



        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buildings_shapefile", type=str, \
                        default='/scratch/metzgern/HAC/data/PopMapData/raw/SwissBuildings/SwissTLM3D/swisstlm3d_2020-03_2056_5728/2020_SWISSTLM3D_SHP_CHLV95_LN02/TLM_BAUTEN/swissTLM3D_TLM_GEBAEUDE_FOOTPRINT.shp', \
                        help="Shapefile with building footprints as polygons")
    parser.add_argument("--target_raster", type=str, default="/scratch/metzgern/HAC/data/PopMapData/merged/EE/che/S2Aspring/che_S2Aspring.tif", help="tif raster that shows the resolution of the output") 
    parser.add_argument("--suffix", type=str, default="s2", help="suffix to add to the output file name")
    args = parser.parse_args()


    process(args.buildings_shapefile, args.target_raster, args.suffix)