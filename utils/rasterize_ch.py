

from os.path import isfile, join, exists 
from os import makedirs, walk, remove
import glob

import pandas as pd
import numpy as np
from osgeo import gdal
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio import CRS
from rasterio.transform import from_origin
import rasterio

from tqdm import tqdm
import scipy
from PIL import Image
from PIL.Image import Resampling as PILRes

tqdm.pandas()
from utils import plot_2dmatrix


# global
cx = 100
cy = 100

# pseudoscaling
ps = 20

def progress_cb(complete, message, cb_data):
    '''Emit progress report in numbers for 10% intervals and dots for 3% in the classic GDAL style'''
    if int(complete*100) % 10 == 0:
        print(f'{complete*100:.0f}', end='', flush=True)
    elif int(complete*100) % 3 == 0:
        print(f'.', end='', flush=True)
        

def rasterize_csv(csv_filename, source_popNN_file, source_popBi_file, template_dir, target_dir, force=False):
    # definition
    resample_alg = gdal.GRA_Cubic

    # read
    if not isfile(source_popBi_file) or force:
        # df = pd.read_csv(csv_filename)[["E_KOORD", "N_KOORD", "B17BTOT"]]
        df = pd.read_csv(csv_filename, sep=";")[["E_KOORD", "N_KOORD", "B20BTOT"]] 

        E_min = df["E_KOORD"].min()
        N_min = df["N_KOORD"].max()-1
        w = ( df["E_KOORD"].max() - df["E_KOORD"].min() )//cx + 1
        h = ( df["N_KOORD"].max() - df["N_KOORD"].min() )//cy + 1
        pop_raster = np.zeros((h,w))

        df["E_IMG"] = (df["E_KOORD"] - E_min) // cx
        df["N_IMG"] = -(df["N_KOORD"] - N_min) // cy

        # pop_raster[df["N_IMG"].tolist(), df["E_IMG"].to_list()] = df["B17BTOT"]
        pop_raster[df["N_IMG"].tolist(), df["E_IMG"].to_list()] = df["B20BTOT"]
        # plot_2dmatrix(pop_raster, vmax=50)

        meta = {"driver": "GTiff", "count": 1, "dtype": "float32", "width":w, "height":h, "crs": CRS.from_epsg(2056),
                "transform": from_origin(E_min, N_min, cx, cy)}
        
        # pop_raster = np.random.rand(*pop_raster.shape)

        with rasterio.open("tmp.tif", 'w', **meta) as dst:
            dst.write(pop_raster,1)

        # resampling to NN and Bicubic
        with rasterio.open("tmp.tif", "r") as src:
            src_crs = src.crs
            transform, width, height = calculate_default_transform(src_crs, "EPSG:3035", src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()

            xres, _, ulx, _, yres, uly = transform[0], transform[1], transform[2], transform[3], transform[4], transform[5] 

            kwargs.update({
                'crs': 3035,
                'transform': rasterio.Affine(xres//ps, 0.0, ulx, 0.0, yres//ps, uly),
                # 'transform': transform,
                'width': width*ps,
                'height': height*ps,
                "compress": "lzw"})
        
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

    class_folders = glob.glob(join(template_dir, "*"))

    for class_path in class_folders:
        filenames = next(walk(class_path), (None, None, []))[2]

        for filename in filenames:
            match_file = join(class_path,filename)
            makedirs(class_path.replace("viirs", "Pop"), exist_ok=True)
            makedirs(class_path.replace("viirs", "PopNN"), exist_ok=True)
            makedirs(class_path.replace("viirs", "PopBi"), exist_ok=True)
            outfile =  match_file.replace("viirs", "Pop")
            outfileNN =  match_file.replace("viirs", "PopNN")
            outfileBi =  match_file.replace("viirs", "PopBi")

            # Prepate the infos from the matchfile
            match_meta = rasterio.open(match_file).meta.copy() 
            xres, _, ulx, _, yres, uly = match_meta["transform"][0], match_meta["transform"][1], match_meta["transform"][2], match_meta["transform"][3], match_meta["transform"][4], match_meta["transform"][5] # waas 

            minx = ulx
            maxy = uly
            maxx = minx + xres * cx # match_raster.RasterXSize
            miny = maxy + yres * cy # match_raster.RasterYSize

            # get indices of the oversized cut
            pmaxy, pminx = reverse_transform *(minx,maxy) 
            pminy, pmaxx = reverse_transform *(maxx,miny)
            pmaxy, pminx = int(pmaxy), int(pminx)
            pminy, pmaxx = int(pminy), int(pmaxx)

            # get the population cut
            popcutNN_upX = popmap_upNN[pminx:pmaxx,pmaxy:pminy] 
            popcutBi_upX = popmap_upBi[pminx:pmaxx,pmaxy:pminy]

            # scale back to regular resolution
            # popcut = scipy.ndimage.zoom(popcutBi_upX, 0.1, order=1)
            popcutNN_up100 = np.array(Image.fromarray(popcutNN_upX).resize(size=(cx, cy), resample=PILRes.NEAREST) )
            popcutBi_up100 = np.array(Image.fromarray(popcutBi_upX).resize(size=(cx, cy), resample=PILRes.NEAREST) )
            popcutBi = np.array(Image.fromarray(popcutBi_upX).resize(size=(cx//10, cy//10), resample=PILRes.NEAREST) )

            # write HR NN to file
            with rasterio.open(outfileNN, "w",  **match_meta) as dst:
                dst.write(popcutNN_up100,1)

            # write HR bilinear to file
            with rasterio.open(outfileBi, "w",  **match_meta) as dst:
                dst.write(popcutBi_up100,1)
            
            # cleanup interpolation artefacts
            popcutBi[popcutBi<1e-4] = 0.0
            
            # write to the downsampled file
            match_meta.update({
                'transform': rasterio.Affine(xres*10, 0.0, ulx, 0.0, yres*10, uly),
                'width': cy//10,
                'height': cx//10}) 
            with rasterio.open(outfile, "w",  **match_meta) as dst:
                dst.write(popcutBi,1)
            
            img = rasterio.open(outfile).read(1)
                    
            print("Done with", outfile)

    return None


def process():
    # source_folder = "/scratch2/metzgern/HAC/data/BFS_CH/2017"
    source_folder = "/scratch2/metzgern/HAC/data/BFS_CH/2020"
    # source_filename = "STATPOP2017.csv"
    source_filename = "STATPOP2020.csv"
    source_meta_poprasterNN = "PopRasterNN.tif"
    source_meta_poprasterBi = "PopRasterBi.tif"
    template_dir = "/scratch/metzgern/HAC/data/So2Sat_POP_Part1/test/00380_22606_zurich/viirs"
    target_dir = "/scratch/metzgern/HAC/data/So2Sat_POP_Part1/test/00380_22606_zurich/Pop"
    target_dirNN = "/scratch/metzgern/HAC/data/So2Sat_POP_Part1/test/00380_22606_zurich/PopNN"
    target_dirBi = "/scratch/metzgern/HAC/data/So2Sat_POP_Part1/test/00380_22606_zurich/PopBi"
    makedirs(target_dir, exist_ok=True)
    makedirs(target_dirNN, exist_ok=True) 
    makedirs(target_dirBi, exist_ok=True) 

    source_file = join(source_folder, source_filename)
    source_popNN_file = join(source_folder, source_meta_poprasterNN)
    source_popBi_file = join(source_folder, source_meta_poprasterBi)
    rasterize_csv(source_file, source_popNN_file, source_popBi_file, template_dir, target_dir, force=False)
    
    return


if __name__=="__main__":
    process()
    print("Done")