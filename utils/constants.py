


# contains all the constants used in the project

import os
import numpy as np
from os.path import dirname

rot_angle = np.arctan(16/97)*(180/np.pi)
img_rows = 100  # patch height
img_cols = 100  # patch width

# inference patch size
inference_patch_size = 1024
overlap = 32

osm_features = 56  # number of osm based features
num_classes = 1  # one for regression

if os.name == "nt":  # locally
    current_dir_path = dirname(dirname((os.getcwd())))

# current_dir_path = "/scratch2/metzgern/HAC/code/So2SatPOP/data"
current_dir_paths = ["/scratch/metzgern/HAC/data",
    "/cluster/work/igp_psr/metzgern/HAC/data",
    "/cluster/scratch/metzgern/HAC/data/",
]
# --info=progress2 --info=name0 --ignore-existing

for name in current_dir_paths:
    if os.path.isdir(name):
        current_dir_path = name
if current_dir_path is None:
    raise Exception("No data folder found")
# current_dir_path = "/cluster/work/igp_psr/metzgern/HAC/data"
# current_dir_path = "/scratch/metzgern/HAC/data"

# paths to So2Sat POP Part1 folder
all_patches_mixed_part1 = os.path.join(current_dir_path, 'So2Sat_POP_Part1')  # path to So2Sat POP Part 1 data folder
all_patches_mixed_train_part1 = os.path.join(all_patches_mixed_part1, 'train')   # path to train folder
all_patches_mixed_test_part1 = os.path.join(all_patches_mixed_part1, 'test')   # path to test folder

# paths to So2Sat POP Part2 folder
all_patches_mixed_part2 = os.path.join(current_dir_path, 'So2Sat_POP_Part2')  # path to So2Sat POP Part 2 data folder
all_patches_mixed_train_part2 = os.path.join(all_patches_mixed_part2, 'train')   # path to train folder
all_patches_mixed_test_part2 = os.path.join(all_patches_mixed_part2, 'test')   # path to test folder

# Sat2Pop data folder
large_file_paths = [
        "/scratch2/metzgern/HAC/data",
        "/scratch/metzgern/HAC/data",
        "/cluster/work/igp_psr/metzgern/HAC/data"
]
for name in large_file_paths:
    if os.path.isdir(name):
        large_file_path = name
if large_file_path is None:
    raise Exception("No data folder found")
pop_map_root = os.path.join(large_file_path, os.path.join("PopMapData", "processed"))
pop_map_root_large = os.path.join("/scratch2/metzgern/HAC/data", os.path.join("PopMapData", "processed"))
pop_map_covariates = os.path.join(large_file_path, os.path.join("PopMapData", os.path.join("merged", "EE")))
pop_map_covariates_large = os.path.join("/scratch2/metzgern/HAC/data", os.path.join("PopMapData", os.path.join("merged", "EE")))


# Sat2Pop data folder
data_paths_aux = [
        "/scratch2/metzgern/HAC/data",
        "/scratch/metzgern/HAC/data",
        "/cluster/work/igp_psr/metzgern/HAC/data",
        # /cluster/work/igp_psr/metzgern/HAC/data/PopMapData/raw/GoogleBuildings
]
for name in data_paths_aux:
    if os.path.isdir(name):
        data_path_aux = name
if large_file_path is None:
    raise Exception("No data folder found")
pop_gbuildings_path = os.path.join(data_path_aux, os.path.join("PopMapData", os.path.join("raw", "GoogleBuildings")))




# Definitions of where to find the census data and the boundaries of the target areas
datalocations = {
    'pricp2': {
        'fine': {
            'boundary': "boundaries4.tif",
            'census': "census4.csv",
        },
        'fineBlockCE': {
            'boundary': "boundaries_BLOCKCE20.tif",
            'census': "census_BLOCKCE20.csv",
        },
        'fineCountyFP': {
            'boundary': "boundaries_COUNTYFP20.tif",
            'census': "census_COUNTYFP20.csv",
        },
        'fineTRACTCE': {
            'boundary': "boundaries_TRACTCE20.tif",
            'census': "census_TRACTCE20.csv",
        },
        'coarse': {
            'boundary': "boundaries_COUNTYFP20.tif",
            'census': "census_COUNTYFP20.csv",
        }
    },
    'rwa': {
        'fine': {
            'boundary': "boundaries_kigali100.tif",
            'census': "census_kigali100.csv",
        },
        'fine100': {
            'boundary': "boundaries_kigali100.tif",
            'census': "census_kigali100.csv",
        },
        'fine200': {
            'boundary': "boundaries_kigali200.tif",
            'census': "census_kigali200.csv",
        },
        'fine400': {
            'boundary': "boundaries_kigali400.tif",
            'census': "census_kigali400.csv",
        },
        'fine500': {
            'boundary': "boundaries_kigali500.tif",
            'census': "census_kigali500.csv",
        },
        'fine1000': {
            'boundary': "boundaries_kigali1000.tif",
            'census': "census_kigali1000.csv",
        },
        'coarse': {
            'boundary': "boundaries_coarse.tif",
            'census': "census_coarse.csv",
        } 
    }
}

testlevels = {
    'pricp2': ["fine", "fineTRACTCE"],
    # 'rwa': ["coarse"]
    'rwa': ["fine100", "fine200", "fine400", "fine1000", "coarse"]
}

    
src_path = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(os.path.join(src_path, 'data'), 'config')
exp_path = os.path.join(os.path.dirname(src_path), 'results')
if not os.path.exists(exp_path):
    os.mkdir(exp_path)

us_city_rasters = os.path.join(current_dir_path, 'city_rasters_us')
all_patches_mixed_us_part1 = os.path.join(current_dir_path, 'US_POP_Part1')
all_patches_mixed_us_part2 = os.path.join(current_dir_path, 'US_POP_Part2')

# all osm features
osm_feature_names = ['aerialway', 'aeroway', 'amenity', 'barrier', 'boundary', 'building', 'craft', 'emergency', 'geological',
'healthcare', 'highway', 'historic', 'landuse', 'leisure', 'man_made', 'military', 'natural',
'office', 'place', 'power', 'public_transport', 'railway', 'route', 'shop', 'sport', 'telecom',
'tourism', 'water', 'waterway', 'addr:housenumber', 'restrictions', 'other', 'n', 'm', 'k_avg',
'intersection_count', 'streets_per_node_avg', 'streets_per_node_counts_argmin',
'streets_per_node_counts_min', 'streets_per_node_counts_argmax', 'streets_per_node_counts_max',
'streets_per_node_proportions_argmin', 'streets_per_node_proportions_min',
'streets_per_node_proportions_argmax', 'streets_per_node_proportions_max', 'edge_length_total',
'edge_length_avg', 'street_length_total', 'street_length_avg', 'street_segment_count',
'node_density_km', 'intersection_density_km', 'edge_density_km', 'street_density_km', 'circuity_avg',
'self_loop_proportion']

# only imp. features for mapping colors in plots
osm_prominent_feature_names = ["edge_length_total", "intersection_count", "k_avg", "street_length_total",
                               "streets_per_node_counts_argmin", "streets_per_node_counts_max", "streets_per_node_counts_min",
                               "streets_per_node_proportions_max", "streets_per_node_proportions_min"]

dpi = 96
fig_size = (400 / dpi, 400 / dpi)
fig_size_heatmap = (400 / dpi, 400 / dpi)

