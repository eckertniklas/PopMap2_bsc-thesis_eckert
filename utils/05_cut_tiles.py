


import argparse
import os
import rasterio


def process(input_path, output_dir, tile_size):

    name = input_path.split("/")[-1]

    # get all the subdirs
    subdirs = [x[0] for x in os.walk(input_path)]

    # Get the first tif file
    subdirs[1]
    template_file = os.path.join( subdirs[1], os.listdir(subdirs[1])[0] ) 

    with rasterio.


    for subdir in subdirs:
        if subdir==input_path:
            continue
        print("Processing files in", subdir)





    return None

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", default="/scratch/metzgern/HAC/data/AfricaSat/merged/EE/pri", type=str, help="processed covariate data")
    parser.add_argument("output_dir", default="/scratch/metzgern/HAC/data/AfricaSat/processed/EE/pri", type=str, help="processed covariate data")
    parser.add_argument("tile_size", default=100, type=int, help="")
    args = parser.parse_args()


    process(args.input_path, args.output_dir, args.tile_size)



if __name__=="__main__":
    main()
    print("Done!")