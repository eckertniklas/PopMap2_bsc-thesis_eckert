
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import shapely.wkt
import argparse
import os
import json
from shapely.geometry import shape
from tqdm import tqdm
tqdm.pandas()
from rasterio.transform import from_origin

from plot import plot_2dmatrix

def rasterize_gbuildings(csv_path, template_path, output_folder):
    """Rasterize the gbuildings dataset.

    Args:
        csv_path (str): path to the csv file
        template_path (str): path to the template raster file
    """

    # Load the buildings data
    df = pd.read_csv(csv_path)

    # Convert the 'geo' column from GeoJSON to a Shapely geometry
    # df['geometry'] = df['.geo'].apply(lambda x: shape(json.loads(x)))
    df['geometry'] = df['.geo'].progress_apply(lambda x: shape(json.loads(x)))

    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Load the template raster again
    with rasterio.open(template_path) as src:
        out_meta = src.meta.copy()



    #### get the segmentations ####
    # Rasterize the polygons
    out_image = rasterize(gdf.geometry, out_shape=(out_meta['height'], out_meta['width']),
                        transform=out_meta['transform'], fill=0, default_value=1)

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[0],
                    "width": out_image.shape[1],
                    "count": 1,
                    "dtype": rasterio.uint8,
                    "compress": 'lzw'})

    # Save the rasterized image
    output_path_segmentation = os.path.join(output_folder, csv_path.split('/')[-1].split('.')[0] + '_segmentation.tif')
    with rasterio.open(output_path_segmentation, "w", **out_meta) as dest:
        dest.write(out_image, 1)
    print(output_path_segmentation)




    #### get the counts ####
    # Calculate the centroids of the polygons
    gdf['centroid'] = gdf.geometry.centroid

    # Create an empty raster to store the counts
    counts = rasterio.open(output_path_segmentation).read(1).copy() * 0

    # Iterate over the centroids
    for point in tqdm(gdf.centroid):
        # Transform the point's coordinates to pixel coordinates
        row, col = ~out_meta['transform'] * (point.x, point.y)
        row, col = int(row), int(col)
        
        # Make sure the point falls within the raster's bounds
        # if (0 <= row < out_meta['height']) and (0 <= col < out_meta['width']):
        if (0 <= row < out_meta['width']) and (0 <= col < out_meta['height']):
            # counts[row, col] += 1
            counts[col, row] += 1
    # Save the count raster
    
    output_path_counts = os.path.join(output_folder, csv_path.split('/')[-1].split('.')[0] + '_counts.tif') 
    with rasterio.open(output_path_counts, "w", **out_meta) as dest:
        dest.write(counts, 1)

        


if __name__ == "__main__":

    # build the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="path to the csv file")
    parser.add_argument("--template_path", type=str, required=True, help="path to the template raster file")
    parser.add_argument("--output_folder", type=str, required=True, help="path to the output raster file")
    args = parser.parse_args()
    
    rasterize_gbuildings(args.csv_path, args.template_path, args.output_folder)
