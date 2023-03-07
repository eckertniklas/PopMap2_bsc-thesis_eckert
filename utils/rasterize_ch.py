

from os.path import isfile, join, exists 
from os import makedirs, walk, remove
import glob

import pandas as pd
import numpy as np
from osgeo import gdal
from rasterio.warp import transform_geom
from rasterio.features import is_valid_geom
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio import CRS
from rasterio.transform import from_origin
import rasterio

from tqdm import tqdm, tqdm_pandas

tqdm.pandas()
from utils import plot_2dmatrix


# global
cx = 100
cy = 100

def progress_cb(complete, message, cb_data):
    '''Emit progress report in numbers for 10% intervals and dots for 3% in the classic GDAL style'''
    if int(complete*100) % 10 == 0:
        print(f'{complete*100:.0f}', end='', flush=True)
    elif int(complete*100) % 3 == 0:
        print(f'.', end='', flush=True)
        

def rasterize_csv(csv_filename, source_pop_file, template_dir, target_dir, force=False):
    # definition
    resample_alg = gdal.GRA_Cubic

    # read
    if not isfile(source_pop_file) or force:
        df = pd.read_csv(csv_filename)[["E_KOORD", "N_KOORD", "B17BTOT"]]

        E_min = df["E_KOORD"].min()
        N_min = df["N_KOORD"].max()-1
        w = ( df["E_KOORD"].max() - df["E_KOORD"].min() )//cx + 1
        h = ( df["N_KOORD"].max() - df["N_KOORD"].min() )//cy + 1
        pop_raster = np.zeros((h,w))

        df["E_IMG"] = (df["E_KOORD"] - E_min) // cx
        df["N_IMG"] = -(df["N_KOORD"] - N_min) // cy

        pop_raster[df["N_IMG"].tolist(), df["E_IMG"].to_list()] = df["B17BTOT"]
        # plot_2dmatrix(pop_raster, vmax=50)

        meta = {"driver": "GTiff", "count": 1, "dtype": "float32", "width":w, "height":h, "crs": CRS.from_epsg(2056),
                "transform": from_origin(E_min, N_min, cx, cy)}

        with rasterio.open(source_pop_file, 'w', **meta) as dst:
            dst.write(pop_raster,1)

        # TODO: check for bugs
        with rasterio.open(source_pop_file) as src:
            src_crs = src.crs
            transform, width, height = calculate_default_transform(src_crs, "EPSG:4326", src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()

            kwargs.update({
                'crs': 4326,
                'transform': transform,
                'width': width,
                'height': height})

            with rasterio.open(source_pop_file, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src,1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=rasterio,
                    resampling=Resampling.nearest)



    # geom = {"type": "Polygon", "coordinates": [[ (x, y) for x,y in zip(df["E_KOORD"], df["N_KOORD"]) ]] }
    # # proj_geom = np.array(transform_geom(CRS.from_epsg(2056), CRS.from_epsg(4326), geom)["coordinates"][0])
    # proj_geom = np.array(transform_geom(CRS.from_epsg(2056), CRS.from_epsg(3035), geom)["coordinates"])[0,:-1]

    # long = proj_geom[:,0]
    # lat = proj_geom[:,1]

    # df["long"] = proj_geom[:,0]
    # df["lat"] = proj_geom[:,1]
    # df.sort_values(['lat','long'])
    # min_long, max_long = df["long"].min(), df["long"].max()
    # min_lat, max_lat = df["lat"].min(), df["lat"].max()

    # df.to_csv("tmp.xyz", index = False, header = None, sep = " ")
    # demn = gdal.Translate("tmp.tif", "tmp.xyz", outputSRS = "EPSG:3035")
    # demn = None
    # remove("tmp.xyz")

    class_folders = glob.glob(join(template_dir, "*"))

    for class_path in class_folders:
        filenames = next(walk(class_path), (None, None, []))[2]

        # open input
        with rasterio.open(source_pop_file) as src:
            src_transform = src.transform

            for filename in filenames:

                match_file = join(class_path,filename)
                makedirs(class_path.replace("viirs", "Pop"), exist_ok=True)
                outfile =  match_file.replace("viirs", "Pop")


                # Prepate the infos from the matchfile
                match_raster = gdal.Open(match_file, gdal.GA_ReadOnly)
                match_prj = match_raster.GetProjection()
                match_prj = match_raster.GetSpatialRef()
                match_tranform = match_raster.GetGeoTransform()
                ulx, xres, xskew, uly, yskew, yres = match_tranform
                height, width = match_raster.RasterXSize, match_raster.RasterYSize # was
                width, height = match_raster.RasterXSize, match_raster.RasterYSize
            
                minx = ulx
                maxy = uly
                maxx = minx + xres * match_raster.RasterXSize
                miny = maxy + yres * match_raster.RasterYSize

                # get the options
                options = gdal.WarpOptions(
                    dstSRS=match_prj,
                    # xRes=xres, yRes=yres,
                    width=width, height=height,
                    # callback=progress_cb,
                    resampleAlg=resample_alg,
                    outputBounds = (minx, miny, maxx, maxy),
                    multithread=True
                    )

                ds = gdal.Warp(
                    destNameOrDestDS=outfile,
                    # srcDSOrSrcDSTab=vrtfile, 
                    srcDSOrSrcDSTab=source_pop_file, 
                    options= options
                )
                

                img = rasterio.open(outfile)
                        
                print("Done with", outfile)


                # with rasterio.open(outfile) as ou: 
                #     img = ou.read(1)
                #     height = ou.height
                #     width = ou.width
                #     this_crs = ou.crs
                #     this_meta = ou.meta.copy()
                #     this_transform = ou.meta["transform"]

                
                # # open input to match
                # with rasterio.open(match_file) as match:
                #     dst_crs = match.crs
                    
                #     # calculate the output transform matrix
                #     dst_transform, dst_width, dst_height = calculate_default_transform(
                #         src.crs,     # input CRS
                #         dst_crs,     # output CRS
                #         match.width,   # input width
                #         match.height,  # input height 
                #         *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
                #     )

                # # set properties for output
                # dst_kwargs = src.meta.copy()
                # dst_kwargs.update({"crs": dst_crs,
                #                 "transform": dst_transform,
                #                 "width": dst_width,
                #                 "height": dst_height,
                #                 "nodata": 0})
                # # print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
                # # open output
                # with rasterio.open(outfile, "w", **dst_kwargs) as dst:
                #     # iterate through bands and write using reproject function
                #     for i in range(1, src.count + 1):
                #         dest, dest_tranform = reproject(
                #             source=rasterio.band(src, i),
                #             destination=rasterio.band(dst, i),
                #             src_transform=src.transform,
                #             src_crs=src.crs,
                #             dst_transform=dst_transform,
                #             dst_crs=dst_crs,
                #             # resampling=Resampling.cubic)
                #             # resampling=Resampling.bilinear)
                #             resampling=Resampling.nearest)
                        
                img = rasterio.open(outfile)
                        
                print("Done with", outfile)
                
    return None


def process():
    source_folder = "/scratch/metzgern/HAC/data/BFS_CH/2017"
    source_filename = "STATPOP2017.csv"
    source_meta_popraster = "PopRaster.tif"
    template_dir = "/scratch/metzgern/HAC/data/So2Sat_POP_Part1/train/00380_22606_zurich/viirs"
    target_dir = "/scratch/metzgern/HAC/data/So2Sat_POP_Part1/train/00380_22606_zurich/Pop"
    makedirs(target_dir, exist_ok=True) 

    source_file = join(source_folder, source_filename)
    source_pop_file = join(source_folder, source_meta_popraster)
    rasterize_csv(source_file, source_pop_file, template_dir, target_dir, force=False)
    
    return


if __name__=="__main__":
    process()
    print("Done")