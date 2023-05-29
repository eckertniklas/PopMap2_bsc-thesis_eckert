

import argparse
import os
from os.path import isfile   
from osgeo import gdal
from tqdm import tqdm

def process(parent_dir, output_dir):

    
    # os.makedirs(parent_dir.replace("raw", "merged"), exist_ok=True)
    name = parent_dir.split("/")[-1]

    # get all subdirectories
    subdirs = [x[0] for x in os.walk(parent_dir)]

    for subdir in subdirs:

        # Sort out fist unusable one
        if subdir==parent_dir:
            continue
        print("Processing files in", subdir)
        
        ty = subdir.split("/")[-1]
        os.makedirs(os.path.join(output_dir, ty), exist_ok=True)

        # Collect files
        files_to_mosaic = [ os.path.join(subdir,file) for file in os.listdir(subdir)]
        
        # Create output name
        # ty = subdir.split("/")[-1]
        output_file = os.path.join(os.path.join(output_dir, ty), name + "_" + ty + ".tif" )

        # Skip if file already exists
        if isfile(output_file):
            print("File already exists, skipping")
            continue
        
        print("Merging files to", output_file)
        # Merge files
        g = gdal.Warp(output_file, files_to_mosaic, format="GTiff", options=["COMPRESS=LZW"]) # if you want
        g = None

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", default="/scratch/metzgern/HAC/data/PopMapData/raw/EE/pri", type=str, help="")
    parser.add_argument("output_dir", default="/scratch/metzgern/HAC/data/PopMapData/processed/EE/pri", type=str, help="")
    args = parser.parse_args()

    process(args.parent_dir, args.output_dir)


if __name__ == "__main__":
    main()
    print("Done!")


