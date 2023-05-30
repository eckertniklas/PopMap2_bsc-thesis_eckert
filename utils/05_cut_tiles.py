


import argparse
import os
import rasterio
from tqdm import tqdm

done = ["S2spring", "S2summer", "S2winter", "S2autumn",
        "S1spring", "S1summer", "S1winter", "S1autumn",
        "VIIRS"]

def process(input_path, output_dir, tile_size=100):
    '''
    Input:
        input_path: path to the directory containing the raw data
        output_dir: path to the directory where the merged data should be stored
        tile_size: size of the tiles in pixels
    Output:
        None   
    '''

    # get all the subdirs
    subdirs = [x[0] for x in os.walk(input_path)]

    # Get the first tif file
    template_file = os.path.join( subdirs[1], os.listdir(subdirs[1])[0] ) 

    # read metadata of the template file
    # this is to make sure that the metadata matches all the other files
    with rasterio.open(template_file, 'r') as tmp:
        metadata = tmp.meta.copy()
        w = metadata["width"]
        h = metadata["height"]

    # Iterate over all subdirs
    for subdir in subdirs:

        # Sort out fist unusable one
        if subdir==input_path:
            continue

        if subdir.split("/")[-1] in done:
            continue

        # Find source file
        source_file = os.path.join( subdir, os.listdir(subdir)[0] ) 
        print("Processing file", source_file)

        # make output dir
        os.makedirs(subdir.replace("raw", "merged"), exist_ok=True)

        # make subdir of output dir
        ty = subdir.split("/")[-1]
        os.makedirs(os.path.join(output_dir, ty), exist_ok=True)
        output_child_dir = os.path.join(output_dir, ty)
        
        # read souce file
        with rasterio.open(source_file, 'r') as src:
            # get the metadata
            metadata = src.meta.copy()

            # loop over the width and height
            file_idx = 0
            for i in tqdm(range(0, w, tile_size)):
                for j in (range(0, h, tile_size)):

                    # get the window
                    window = rasterio.windows.Window(i, j, tile_size, tile_size)

                    # save it into a new file an enumerate it
                    output_file = os.path.join(output_child_dir, str(file_idx) + "_" + ty + "_" + ".tif")

                    # modify the metadata
                    metadata.update({
                        'height': window.height,
                        'width': window.width,
                        'transform': rasterio.windows.transform(window, src.transform),
                        'count': src.count
                    })
                    
                    # read the window from the source file and write it to the new file
                    with rasterio.open(output_file, 'w', **metadata) as dst:
                        dst.write(src.read(window=window))
                    file_idx += 1

    return None

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", default="/scratch/metzgern/HAC/data/PopMapData/merged/EE/pri", type=str, help="processed covariate data")
    parser.add_argument("output_dir", default="/scratch/metzgern/HAC/data/PopMapData/processed/EE/pri", type=str, help="processed covariate data")
    parser.add_argument("tile_size", default=100, type=int, help="")
    args = parser.parse_args()


    process(args.input_path, args.output_dir, args.tile_size)



if __name__=="__main__":
    main()
    print("Done!")