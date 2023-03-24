
import argparse
import requests

import ee
try:
    ee.Initialize()
except:
    print("couldn't init EE")
    ee.Authenticate(auth_mode="localhost")
    ee.Initialize()
    # gcloud auth application-default login --no-browser

ee_crs = ee.Projection('EPSG:4326')
    
# Sentinel 2 Config
# Sen2spring_start_date = '2019-03-01'
# Sen2spring_finish_date = '2019-06-01'
# Sen2summer_start_date = '2019-06-01'
# Sen2summer_finish_date = '2019-09-01'
# Sen2autumn_start_date = '2019-09-01'
# Sen2autumn_finish_date = '2019-12-01'
# Sen2winter_start_date = '2019-12-01'
# Sen2winter_finish_date = '2020-03-01'

# Sentinel 2 Config
Sen2spring_start_date = '2020-03-01'
Sen2spring_finish_date = '2020-06-01'
Sen2summer_start_date = '2020-06-01'
Sen2summer_finish_date = '2020-09-01'
Sen2autumn_start_date = '2020-09-01'
Sen2autumn_finish_date = '2020-12-01'
Sen2winter_start_date = '2020-12-01'
Sen2winter_finish_date = '2021-03-01'

# AOI = ee.Geometry.Point(-122.269, 45.701)
# START_DATE = '2020-06-01'
# END_DATE = '2020-09-01'
CLOUD_FILTER = 60
CLD_PRB_THRESH = 60
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 60

Sentinel1_start_date = '2020-07-03'
Sentinel1_finish_date = '2020-08-30'
orbit = 'DESCENDING' # Default
# orbit = 'Ascending' # for pri?


# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
def get_s2_sr_cld_col(aoi, start_date, end_date):
    """Join Sentinel-2 Surface Reflectance and Cloud Probability
    This function retrieves and joins ee.ImageCollections:
    'COPERNICUS/S2_SR' and 'COPERNICUS/S2_CLOUD_PROBABILITY'
    Parameters
    ----------
    aoi : ee.Geometry or ee.FeatureCollection
      Area of interested used to filter Sentinel imagery
    params : dict
      Dictionary used to select and filter Sentinel images. Must contain
      START_DATE : str (YYYY-MM-DD)
      END_DATE : str (YYYY-MM-DD)
      CLOUD_FILTER : int
        Threshold percentage for filtering Sentinel images
    """
        
    # Import and filter S2 SR.
    # s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')  # Default
    # s2_sr_col = (ee.ImageCollection('COPERNICUS/S2') 
    
    # Real Data from raw S2 collection
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2') 
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    
    # Query the Sentinel-2 SR collection with cloud masks.
    s2_sr_col_FORMASKS = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') 
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
        .select('SCL'))
        
    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))
    # print(s2_cloudless_col.getInfo()["bands"])

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    # return s2_sr_col
    merge1 =  ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

    #  merge1 with s2_sr_col_FORMASKS
    # merge1 =  ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
    #     'primary': merge1,
    #     'secondary': s2_sr_col_FORMASKS,
    #     'condition': ee.Filter.equals(**{
    #         'leftField': 'system:index',
    #         'rightField': 'system:index'
    #     })
    # }))

    # merge1 = merge1.addBands(s2_sr_col_FORMASKS)

    # Combine the two collections merge1 and s2_sr_col_FORMASKS

    merge1 = ee.ImageCollection.combine(merge1, s2_sr_col_FORMASKS)

    return merge1


# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
def add_cloud_bands(img):
    """Add cloud bands to Sentinel-2 image
    Parameters
    ----------
    img : ee.Image
      Sentinel 2 image including (cloud) 'probability' band
    params : dict
      Parameter dictionary including
      CLD_PRB_THRESH : int
        Threshold percentage to identify cloudy pixels
    """
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
def add_shadow_bands(img):
    """Add cloud shadow bands to Sentinel-2 image
    Parameters
    ----------
    img : ee.Image
      Sentinel 2 image including (cloud) 'probability' band
    params : dict
      Parameter dictionary including
      NIR_DRK_THRESH : int
        Threshold percentage to identify potential shadow pixels as dark pixels from NIR band
      CLD_PRJ_DIST : int
        Distance to project clouds along azimuth angle to detect potential cloud shadows
    """
    
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    # return img.addBands(is_cld_shdw)
    # return img_cloud_shadow.addBands(is_cld_shdw)
    return img.addBands(is_cld_shdw)


# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


def submit_s2job(s2_sr_mosaic, description, name, exportarea):
    # Submits a job to Google earth engine with all the requred arguments
    
    task = ee.batch.Export.image.toDrive(
        image = s2_sr_mosaic,
        scale = 10,  
        description = description + "_" + name,
        fileFormat="GEOTIFF", 
        folder = name, 
        region = exportarea,
        crs='EPSG:4326',
        maxPixels=80000000000 
    )

    # submit/start the job
    task.start() 

def download(minx, miny, maxx, maxy, name):

    exportarea = { "type": "Polygon",  "coordinates": [[[maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny], [maxx, miny]]]  }
    exportarea = ee.Geometry.Polygon(exportarea["coordinates"]) 

    S1 = True
    S2 = True
    VIIRS = True

    if S1:
        ########################### Processing Sentinel 1 #############################################
        
        # select by data and sensormode and area
        collectionS1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
            .filter(ee.Filter.eq('orbitProperties_pass', orbit))\
            .filterBounds(exportarea)\
            .filterDate(Sentinel1_start_date, Sentinel1_finish_date)\
            .select(['VV', 'VH'])
        
        # Reduce with Median operation
        collectionS1_mean = collectionS1.mean()
        # collectionS1_median = collectionS1.first()
        # collectionS1_median = collectionS1.mosaic()
        # collectionS1_median = collectionS1.mean()

        # Export
        task = ee.batch.Export.image.toDrive(
                        image = collectionS1_mean,
                        scale = 10,  
                        description = "S1_" + name,
                        fileFormat="GEOTIFF", 
                        folder = name, 
                        region = exportarea,
                        crs='EPSG:4326',
                        maxPixels=80000000000,
                    )
        task.start()

    if S2:
        ########################### Processing Sentinel 2 #############################################
        # 1. cating the clouds to the Sentinel-2 data
        # 2. Filtering clouds and cloud shadow and apply the mask to sentinel-2
        # 3. composite the image by giving preference to the least cloudy image first.
        # 4. Submit job

        # SPRING
        s2_sr_cld_col = get_s2_sr_cld_col(exportarea, Sen2spring_start_date, Sen2spring_finish_date)
        s2_sr_col = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)
        s2_sr_col = s2_sr_col.sort('CLOUDY_PIXEL_PERCENTAGE', False)
        s2_sr_col = s2_sr_col.mosaic()
        submit_s2job(s2_sr_col, "sen2spring",  name, exportarea)

        # SUMMER
        s2_sr_cld_col = get_s2_sr_cld_col(exportarea, Sen2summer_start_date, Sen2summer_finish_date)
        s2_sr_col = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)
        s2_sr_col = s2_sr_col.sort('CLOUDY_PIXEL_PERCENTAGE', False).mosaic()
        submit_s2job(s2_sr_col, "sen2summer", name, exportarea)

        # AUTUMN
        s2_sr_cld_col = get_s2_sr_cld_col(exportarea, Sen2autumn_start_date, Sen2autumn_finish_date)
        s2_sr_col = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)
        s2_sr_col = s2_sr_col.sort('CLOUDY_PIXEL_PERCENTAGE', False).mosaic()
        submit_s2job(s2_sr_col, "sen2autumn", name, exportarea)

        # WINTER
        s2_sr_cld_col = get_s2_sr_cld_col(exportarea, Sen2winter_start_date, Sen2winter_finish_date)
        s2_sr_col = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)
        s2_sr_col = s2_sr_col.sort('CLOUDY_PIXEL_PERCENTAGE', False).mosaic()
        submit_s2job(s2_sr_col, "sen2winter", name, exportarea)
        # print(s2_sr_median.getInfo()["bands"])

    if VIIRS:
        ########################### Processing Sentinel 2 #############################################

        viirs_NL_col = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG') \
                    .filter(ee.Filter.date(Sen2spring_start_date, Sen2winter_finish_date))
        NL_median = viirs_NL_col.select("avg_rad").median()

        # Create composite
        # num_obs = viirs_NL_col.select("cf_cvg")
        # NL_col = NL_col.updateMask(num_obs.gt(1))
        # viirs_NL = viirs_NL_col.mosaic()
        
        # Export
        task = ee.batch.Export.image.toDrive(
                        image = NL_median,
                        scale = 10,  
                        description = "VIIRS_" + name,
                        fileFormat="GEOTIFF", 
                        folder = name, 
                        region = exportarea,
                        crs='EPSG:4326',
                        maxPixels=80000000000,
                    )
        task.start()

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("minx", type=float)
    parser.add_argument("miny", type=float)
    parser.add_argument("maxx", type=float)
    parser.add_argument("maxy", type=float) 
    parser.add_argument("name", type=str) 
    parser.add_argument("output_path", type=str, help="Output path")
    args = parser.parse_args()

    download(args.minx, args.miny, args.maxx, args.maxy, args.name)


if __name__ == "__main__":
    main()
    print("Done!")


