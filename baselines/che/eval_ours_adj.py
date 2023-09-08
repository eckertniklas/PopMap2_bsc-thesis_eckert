import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')

# from ..data.PopulationDataset_target import Population_Dataset_target
from data.PopulationDataset_target import Population_Dataset_target
import rasterio
from rasterio.warp import transform_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
from utils.metrics import get_test_metrics
import torch
from utils.plot import plot_2dmatrix, plot_and_save, scatter_plot3


def reproject_maps(map_path, template_path, output_path, sumpool=False):

    # get transform of original map
    with rasterio.open(map_path) as src:
        src_transform = src.transform

    # translate the the meta maps in the template file coordinate system
    with rasterio.open(map_path) as src:
        # Open the fine resolution raster
        with rasterio.open(template_path) as dst:
            
            # Transform the bounds of the destination raster to the source crs
            dst_bounds = transform_bounds(dst.crs, src.crs, *dst.bounds)

            # Calculate the transformation matrix from the source (coarse) to destination (fine) crs
            transform, width, height = calculate_default_transform(src.crs, dst.crs, src.width, src.height, *dst_bounds,
                                                                    dst_width=dst.width, dst_height=dst.height)

            # Create a new dataset to store the reprojected raster
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst.crs,
                # 'transform': transform,
                'transform': dst.transform,
                'width': width,
                'height': height
            })

            # Write the reprojected raster to the new dataset
            with rasterio.open(output_path, 'w', **kwargs) as reproj:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(reproj, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst.crs,
                        resampling=Resampling.sum if sumpool else Resampling.nearest
                    )

    x_stretch = src_transform.a / dst.transform.a
    y_stretch = src_transform.e / dst.transform.e
    
    print("Reprojected map saved to: ", output_path)
    return x_stretch, y_stretch



def evaluate_meta_maps(map_path, template_path, wpop_raster_template):

    parent_dir = "/".join(map_path.split("/")[:-1])

    # high_resolution map 
    hr_map_path = map_path.replace(".tif", "_hr.tif")
    # hr_map_path = map_path.replace(".tiff", "_hr.tiff")

    # load map
    with rasterio.open(map_path) as src:
        hr_pop_map = torch.from_numpy(src.read(1))

    # replace nan values with 0
    hr_pop_map[hr_pop_map != hr_pop_map] = 0
    hr_pop_map[hr_pop_map < 0] = 0

    
    # define GT dataset
    dataset = Population_Dataset_target("che", train_level="coarse4") 

    # adjust map with the coarse census
    hr_pop_map_adj = dataset.adjust_map_to_census(hr_pop_map.clone())

    # save adjusted map
    hr_map_path_adj = map_path.replace(".tif", "_hr_adj.tif")
    metadata = src.meta.copy()
    metadata.update({"dtype": "float32",
                     "compress": "lzw"})

    if not os.path.exists( hr_map_path_adj):
        with rasterio.open(hr_map_path_adj, 'w', **metadata) as dst:
            dst.write(hr_pop_map_adj.to(torch.float32).cpu().numpy(), 1)
        print("Adjusted map saved to: ", hr_map_path_adj)
    
    # reproject to the worldpop map
    hr_map_path_reproj = map_path.replace(".tif", "_hr_reproj.tif")
    if not os.path.exists(hr_map_path_reproj):
        _, _ = reproject_maps(map_path, wpop_raster_template, hr_map_path_reproj, sumpool=True)
        print("Reprojected map to worldpop raster saved to: ", hr_map_path_reproj)
    else:
        print("Reprojected map to worldpop raster already exists: ", hr_map_path_reproj)

    # reproject to the worldpop map
    hr_map_path_adj_reproj = map_path.replace(".tif", "_hr_adj_reproj.tif")
    if not os.path.exists(hr_map_path_adj_reproj):
        _, _ = reproject_maps(hr_map_path_adj, wpop_raster_template, hr_map_path_adj_reproj, sumpool=True)
        print("Reprojected adjusted map to worldpop raster saved to: ", hr_map_path_adj_reproj)
    else:
        print("Reprojected adjusted map to worldpop raster already exists: ", hr_map_path_adj_reproj)

    # define levels
    # levels = ["finezurich", "finezurich2", "coarse"]
    # levels = ["coarse"]
    levels = ["fine", "coarse", "finezurich", "finezurich2"]

    for level in levels:
        print("Evaluating level: ", level)
        print("-------------------------------")
        print("Direct metrics:")
        census_pred, census_gt = dataset.convert_popmap_to_census(hr_pop_map, gpu_mode=True, level=level)
        test_metrics_meta = get_test_metrics(census_pred, census_gt.float().cuda() )
        print(test_metrics_meta)

        scatterplot = scatter_plot3(census_pred.tolist(), census_gt.tolist())
        scatterplot.save(os.path.join(parent_dir, "last_scatter_direct_{}.png".format(level)))
        
        print("-------------------------------")
        print("Adjusted metrics:")
        census_pred_adj, census_gt = dataset.convert_popmap_to_census(hr_pop_map_adj, gpu_mode=True, level=level)
        test_metrics_meta_adj = get_test_metrics(census_pred_adj, census_gt.float().cuda() )
        print(test_metrics_meta_adj)

        scatterplot_adj = scatter_plot3(census_pred_adj.tolist(), census_gt.tolist())
        scatterplot_adj.save(os.path.join(parent_dir, "last_scatter_adj_{}.png".format(level)))

        print("---------------------------------")

    print("Done")

    pass


if __name__=="__main__":
    """
    Evaluates our-maps on the test set of Switz
    """
    # sample
    # map_path = "/scratch2/metzgern/HAC/POMELOv2_results/So2Sat/experiment_1599_252/rwa_predictions.tif"
    # map_path = "/scratch2/metzgern/HAC/POMELOv2_results/So2Sat/experiment_1610_250/rwa_predictions.tif"
    map_path = "/scratch2/metzgern/HAC/POMELOv2_results/euler/experiment_294_85/che_predictions.tif"
    # map_path = "/scratch2/metzgern/Downloads/experiment_155_459/rwa_predictions.tif"
    # map_path = "/scratch2/metzgern/Downloads/experiment_155_459/rwa_predictions.tif"
    # map_path = "/scratch2/metzgern/HAC/POMELOv2_results/So2Sat/experiment_1542_618/rwa_predictions.tif"


    template_path = "/scratch2/metzgern/HAC/data/PopMapData/merged/EE/che/S2Aautumn/che_S2Aautumn.tif"
    wpop_raster_template = "/scratch2/metzgern/HAC/data/PopMapData/raw/WorldPopMaps/CHE/che_ppp_2020_constrained.tif"

    evaluate_meta_maps(map_path, template_path, wpop_raster_template)
