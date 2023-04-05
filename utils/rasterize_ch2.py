

from os.path import isfile, join, exists 
from os import makedirs, walk, remove
import glob

import pandas as pd
import numpy as np
from osgeo import gdal
from rasterio.warp import transform_geom
from rasterio.features import is_valid_geom
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.windows import Window
from rasterio import CRS
from rasterio.transform import from_origin
import rasterio

from tqdm import tqdm, tqdm_pandas
import scipy
from PIL import Image
from PIL.Image import Resampling as PILRes

tqdm.pandas()
from utils.plot import plot_2dmatrix



def progress_cb(complete, message, cb_data):
    '''Emit progress report in numbers for 10% intervals and dots for 3% in the classic GDAL style'''
    if int(complete*100) % 10 == 0:
        print(f'{complete*100:.0f}', end='', flush=True)
    elif int(complete*100) % 3 == 0:
        print(f'.', end='', flush=True)
        

def rasterize_csv(csv_filename, source_popNN_file, source_popBi_file, template_file, output_file, force=False):
    # definition
    resample_alg = gdal.GRA_Cubic

    # global
    cx = 100
    cy = 100

    # pseudoscaling
    ps = 20


    # with rasterio.open(template_file, "r") as tmp:
    #     tmp_meta = tmp.meta.copy()

    # read
    if not isfile(source_popBi_file) or force:

        # read swiss census data
        df = pd.read_csv(csv_filename)[["E_KOORD", "N_KOORD", "B17BTOT"]]

        E_min = df["E_KOORD"].min()
        N_min = df["N_KOORD"].max()-1
        w = ( df["E_KOORD"].max() - df["E_KOORD"].min() )//cx + 1
        h = ( df["N_KOORD"].max() - df["N_KOORD"].min() )//cy + 1
        pop_raster = np.zeros((h,w))

        df["E_IMG"] = (df["E_KOORD"] - E_min) // cx
        df["N_IMG"] = -(df["N_KOORD"] - N_min) // cy

        # convert to raster
        pop_raster[df["N_IMG"].tolist(), df["E_IMG"].to_list()] = df["B17BTOT"]
        # plot_2dmatrix(pop_raster, vmax=50)

        meta = {"driver": "GTiff", "count": 1, "dtype": "float32", "width":w, "height":h, "crs": CRS.from_epsg(2056),
                "transform": from_origin(E_min, N_min, cx, cy)}

        # save it as temp raster
        with rasterio.open("tmp.tif", 'w', **meta) as dst:
            dst.write(pop_raster,1)

        # resampling to NN and Bicubic with a much larger resolution to avoid resampling artefacts
        with rasterio.open("tmp.tif", "r") as src:
            src_crs = src.crs
            transform, width, height = calculate_default_transform(src_crs, "EPSG:4326", src.width, src.height, *src.bounds)
            # transform, width, height = calculate_default_transform(src_crs, tmp_meta["crs"], src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()

            xres, _, ulx, _, yres, uly = transform[0], transform[1], transform[2], transform[3], transform[4], transform[5] 

            kwargs.update({
                # 'crs': tmp_meta["crs"],
                'crs': 4326,
                'transform': rasterio.Affine(xres/ps, 0.0, ulx, 0.0, yres/ps, uly),
                # 'transform': transform,
                'width': width*ps,
                'height': height*ps})

            # begin reproject to wgs84
            with rasterio.open(source_popNN_file, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src,1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=rasterio,
                    resampling=Resampling.nearest)
                
            with rasterio.open(source_popBi_file, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src,1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=rasterio,
                    resampling=Resampling.bilinear)
                
        remove("tmp.tif")

    # read previously save tifs to disk
    with rasterio.open(source_popNN_file, "r") as src:
        src_transform = src.transform
        src_meta = src.meta.copy()
        reverse_transform = ~src_transform
        popmap_upNN = src.read(1)

    with rasterio.open(source_popBi_file, "r") as src:
        src_transform = src.transform
        src_meta = src.meta.copy()
        reverse_transform = ~src_transform
        popmap_upBi = src.read(1)

    # cut into the template shape that is also in wgs84
    tmp_meta = rasterio.open(template_file).meta.copy() 
    xres, _, ulx, _, yres, uly = tmp_meta["transform"][0], tmp_meta["transform"][1], tmp_meta["transform"][2], tmp_meta["transform"][3], tmp_meta["transform"][4], tmp_meta["transform"][5]

    minx = ulx
    maxy = uly
    maxx = minx + xres * tmp_meta["width"] # match_raster.RasterXSize is the width in this coord system
    miny = maxy + yres * tmp_meta["height"] # match_raster.RasterYSize is the height in this coord system

    # get indices of the oversized cut
    pmaxy, pminx = reverse_transform *(minx,maxy) 
    pminy, pmaxx = reverse_transform *(maxx,miny)
    pmaxy, pminx = int(pmaxy), int(pminx)
    pminy, pmaxx = int(pminy), int(pmaxx)

    # check if padding is needed
    if pminx<0:
        pad = -pminx
        pminx = 0

    # get the population cut
    popcutNN_upX = popmap_upNN[pminx:pmaxx,pmaxy:pminy] 
    popcutBi_upX = popmap_upBi[pminx:pmaxx,pmaxy:pminy]

    # ugly padding
    popcutNN_upX = np.pad(popcutNN_upX, ((pad,0),(0,0)))
    popcutBi_upX = np.pad(popcutBi_upX, ((pad,0),(0,0)))

    # read and resample to resolution of the template
    popcutBi = np.array(Image.fromarray(popcutBi_upX).resize(size=(tmp_meta["width"], tmp_meta["height"]), resample=PILRes.NEAREST) )
    
    # cleanup interpolation artefacts
    popcutBi[popcutBi<1e-4] = 0.0
    
    with rasterio.open(output_file, "w",  **tmp_meta) as dst:
        dst.write(popcutBi,1)
    
    #test
    img = rasterio.open(output_file).read(1)       
    print("Done with", output_file)

    return None


def process():
    # inputs
    source_folder = "/scratch/metzgern/HAC/data/BFS_CH/2017"
    source_filename = "STATPOP2017.csv"
    template_file = "/scratch2/metzgern/HAC/data/Covariates/CHE/Accessibility/che_tt50k_100m_2000.tif"

    # ouputs in the source_folder
    source_meta_poprasterNN = "PopRasterNN2.tif"
    source_meta_poprasterBi = "PopRasterBi2.tif"
    output_file = "ouput_file.tif"

    source_file = join(source_folder, source_filename)
    source_popNN_file = join(source_folder, source_meta_poprasterNN)
    source_popBi_file = join(source_folder, source_meta_poprasterBi)
    output_file = join(source_folder, output_file)

    rasterize_csv(source_file, source_popNN_file, source_popBi_file, template_file, output_file, force=True)
    
    return


if __name__=="__main__":
    process()
    print("Done")