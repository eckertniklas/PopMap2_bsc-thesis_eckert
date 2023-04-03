

import argparse
import geopandas as gdp
import numpy as np
import rasterio
from rasterio import features
import os
import torch
from tqdm import tqdm

from utils import plot_2dmatrix

def process(sh_path, output_tif_file, output_census_file, template_file, gpu_mode=True):

    # read shapefile
    gdb = gdp.read_file(sh_path)

    gdb["idx"] = np.arange(len(gdb))

    # len(np.unique(gdb["COUNTYFP20"])) #78
    # len(np.unique(gdb["BLOCKCE20"])) #363
    # len(np.unique(gdb["TRACTCE20"])) #952
    # len(np.unique(gdb["GEOID20"])) #41987



    # read metadata of the template file
    with rasterio.open(template_file, 'r') as tmp:
        metadata = tmp.meta.copy()

    # make sure that the metadata matches
    metadata.update({"count": 1})

    this_outputfile = output_tif_file.replace("boundaries", "boundaries4")
    this_censusfile = output_census_file.replace("census", "census4")

    # rasterize
    with rasterio.open(this_outputfile, 'w+', **metadata) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,i) for i, geom in zip(gdb["idx"], gdb.geometry))

        # flattens the shapefile into the raster (burns them in)
        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)

    # bring to torch for parallel implementation
    burned = torch.tensor(burned, dtype=torch.int32)
    if gpu_mode:
        burned = burned.cuda()

    gdb["bbox"] = ""
    gdb["count"] = 0
    
    # precalculate all the bounding boxes
    for i in tqdm(gdb["idx"]):
        mask = burned==i
        count = mask.sum()
        if count==0:
            xmin, xmax = 0, 0
            ymin, ymax = 0, 0
        else:
            vertical_indicies = torch.where(torch.any(mask,dim=1))[0]
            horizontal_indicies = torch.where(torch.any(mask,dim=0))[0]
            xmin, xmax = vertical_indicies[[0, -1]].cpu()
            ymin, ymax = horizontal_indicies[[0, -1]].cpu()
            xmax, ymax = xmax+1, ymax+1
            xmin, xmax, ymin, ymax = xmin.item(), xmax.item(), ymin.item(), ymax.item()

        gdb.at[i,"bbox"] = (xmin, xmax, ymin, ymax)
        gdb.loc[i,"count"] = count.cpu().item()

        # test
        # p = 1
        # plot_2dmatrix(burned[xmin-p:xmax+p, ymin-p:ymax+p])

    # write censusdata
    gdb[["idx", "POP20", "bbox", "count"]].to_csv(this_censusfile)

    # len(np.unique(gdb["COUNTYFP20"])) #78
    # len(np.unique(gdb["BLOCKCE20"])) #363
    # len(np.unique(gdb["TRACTCE20"])) #952
    # len(np.unique(gdb["GEOID20"])) #41987

    levels = ["COUNTYFP20", "BLOCKCE20", "TRACTCE20"]
       
    for level in levels:

        thisdb = gdb[["POP20", "geometry",level]].dissolve(by=level, aggfunc=np.sum).reset_index()
        thisdb["idx"] = np.arange(len(thisdb))

        this_outputfile = output_tif_file.replace("boundaries", "boundaries_" + level)
        this_censusfile = output_census_file.replace("census", "census_" + level)


        with rasterio.open(this_outputfile, 'w+', **metadata) as out:
            out_arr = out.read(1)

            # this is where we create a generator of geom, value pairs to use in rasterizing
            shapes = ((geom,i) for i, geom in zip(thisdb["idx"], thisdb.geometry))

            # flattens the shapefile into the raster (burns them in)
            burned1 = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
            out.write_band(1, burned1)

        # bring to torch for parallel implementation
        burned1 = torch.tensor(burned1, dtype=torch.int32)
        if gpu_mode:
            burned1 = burned1.cuda()

        thisdb["bbox"] = ""
        thisdb["count"] = 0
        
        # precalculate all the bounding boxes
        for i in tqdm(thisdb["idx"]):
            mask = burned1==i
            count = mask.sum()
            if count==0:
                xmin, xmax = 0, 0
                ymin, ymax = 0, 0
            else:
                # get the bounding box by calculating the min and max of the indices
                vertical_indicies = torch.where(torch.any(mask,dim=1))[0]
                horizontal_indicies = torch.where(torch.any(mask,dim=0))[0]
                xmin, xmax = vertical_indicies[[0, -1]].cpu()
                ymin, ymax = horizontal_indicies[[0, -1]].cpu()
                xmax, ymax = xmax+1, ymax+1
                xmin, xmax, ymin, ymax = xmin.item(), xmax.item(), ymin.item(), ymax.item()

            thisdb.at[i,"bbox"] = (xmin, xmax, ymin, ymax)
            thisdb.loc[i,"count"] = count.cpu().item()

            # test
            # p = 1
            # plot_2dmatrix(burned[xmin-p:xmax+p, ymin-p:ymax+p])

        # write censusdata
        thisdb[["idx", "POP20", "bbox", "count"]].to_csv(this_censusfile)

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sh_path", default="/scratch/metzgern/HAC/data/PopMapData/raw/boundaries/pri2017", type=str, help="Shapefile with boundaries and census")
    parser.add_argument("output_tif_file", default="/scratch/metzgern/HAC/data/PopMapData/processed/pri2017/boundaries.tif", type=str, help="")
    parser.add_argument("output_census_file", default="/scratch/metzgern/HAC/data/PopMapData/processed/pri2017/census.csv", type=str, help="")
    parser.add_argument("template_file", default="/scratch/metzgern/HAC/data/PopMapData/raw/EE/pri/S1/pri_S1.tif", type=str, help="")
    args = parser.parse_args()

    process(args.sh_path, args.output_tif_file, args.output_census_file, args.template_file, gpu_mode=True)


if __name__ == "__main__":
    main()
    print("Done!")






