

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
        else:
            print("Merging files to", output_file)
            # Merge files
            g = gdal.Warp(output_file, files_to_mosaic, format="GTiff", options=["COMPRESS=LZW"]) # if you want
            g = None

        if ty in ["S2spring", "S2summer", "S2autumn", "S2winter"]:
            # Create same file but only keep bands 2,3,4,8
            ty1C = ty.replace("S2", "S21C")
            output_file1C = output_file.replace("S2", "S21C")
            output_dir1C = output_dir.replace("S2", "S21C")
            os.makedirs(os.path.join(output_dir1C, ty1C), exist_ok=True)

            if isfile(output_dir1C):
                print("File already exists, skipping")
                continue
            
            # g = gdal.Warp(output_file, files_to_mosaic, format="GTiff", options=["COMPRESS=LZW", "BANDS=2,3,4,8"]) # if you want 
            # g = None

            print("Cutting files to", output_file)
            # load file and only save bands 2,3,4,8
            g = gdal.Open(output_file)
            g = gdal.Translate(output_file1C, g, format="GTiff", bandList=[2,3,4,8], options=["COMPRESS=LZW"])
            g = None

s     return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", default="/scratch/metzgern/HAC/data/PopMapData/raw/EE/pri", type=str, help="")
    parser.add_argument("output_dir", default="/scratch/metzgern/HAC/data/PopMapData/processed/EE/pri", type=str, help="")
    args = parser.parse_args()

    process(args.parent_dir, args.output_dir)


if __name__ == "__main__":
    main()
    print("Done!")


