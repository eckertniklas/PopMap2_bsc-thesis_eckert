
import argparse
import requests
import time
import ee
import os



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

def start(task):
        # Start the task.
    try:
        task.start()
    except ee.ee_exception.EEException:
        for i in range(128):
            print("Congrats. too-many jobs. EE is at it's limit. Trial", i,". Taking a 15s pause...")
            time.sleep(15)
            try:
                task.start()
            except:
                pass
            else:
                break 
            if i>30:
                raise Exception("Could not submit EE job")
            


def download_tile(url, filename, folder):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder, filename), 'wb') as f:
            f.write(response.content)
    else:
        print(f"Error downloading {filename}: {response.status_code}")


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
        image=s2_sr_mosaic,
        scale=10,  
        description=description + "_" + name,
        fileFormat="GEOTIFF", 
        folder=name, 
        region=exportarea,
        crs='EPSG:4326', #OLD
        # crs='EPSG:3035', #NEW, but wrong, this only works for Europe
        maxPixels=80000000000 
    )

    # submit/start the job
    task.start() 


# New function to download the Sentine2 data
def export_cloud_free_sen2(season, dates, roi_id, roi, debug=0, S2type="S2"):
    """
    Export cloud free Sentinel-2 data for a given season and region of interest
    Parameters
    ----------
    season : str
        Season to download data for
    dates : list
        List of dates to download data for
    roi_id : str
        Region of interest ID
    roi : ee.Geometry
        Region of interest
    debug : int
        Debug level
    -------
    Returns
        None
    """

    if S2type == "S2":
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
    elif S2type == "S2_SR_HARMONIZED":
        bands = ['B2', 'B3', 'B4', 'B8']
    
    # Get the start and end dates for the season.
    start_date = ee.Date(dates[0])
    end_date = ee.Date(dates[1])

    # Get the Sentinel-2 surface reflectance and cloud probability collections.
    s2_sr = ee.ImageCollection("COPERNICUS/" + S2type)
    s2_clouds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")

    # Filter the collections by the ROI and date.
    criteria = ee.Filter.And(ee.Filter.bounds(roi), ee.Filter.date(start_date, end_date))

    # Filter the collections by the ROI and date.
    s2_sr = s2_sr.filter(criteria)
    s2_clouds = s2_clouds.filter(criteria)

    # Join S2 SR with cloud probability dataset to add cloud mask.
    join = ee.Join.saveFirst('cloud_mask')
    condition = ee.Filter.equals(leftField='system:index', rightField='system:index')
    s2_sr_with_cloud_mask = join.apply(primary=s2_sr, secondary=s2_clouds, condition=condition)

    # Define a function to mask clouds using the probability threshold.
    def mask_clouds(img):
        clouds = ee.Image(img.get('cloud_mask')).select('probability')
        is_not_cloud = clouds.lt(65)
        return img.updateMask(is_not_cloud)

    # Map the function over one year of data and take the median.
    img_c = ee.ImageCollection(s2_sr_with_cloud_mask).map(mask_clouds)

    # Get the median of each pixel for the time period.
    cloud_free = img_c.median()
    # filename = f"{roi_id}_{season}"
    filename = f"{season}_{roi_id}"

    
    # Export the image to Google Drive.
    task = ee.batch.Export.image.toDrive(
        image=cloud_free.select(bands),
        description=filename,
        scale=10,
        region=roi,
        folder=f"{roi_id}/{season}",
        fileNamePrefix=filename,
        maxPixels=1e13
    )

    task.start()

def export_S1_tile(season, dates, filename, roi, folder, scale=10, crs='EPSG:4326', url_mode=True):
    """
    Export Sentinel-1 data for a given season and region of interest
    """
    start_date = ee.Date(dates[0])
    end_date = ee.Date(dates[1])

    # Define a method for filtering and compositing.
    collectionS1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    collectionS1 = collectionS1.filter(ee.Filter.eq('instrumentMode', 'IW'))
    collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    collectionS1 = collectionS1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    collectionS1 = collectionS1.filterBounds(roi)
    # collectionS1 = collectionS1.filter(ee.Filter.contains('.geo', roi))
    collectionS1 = collectionS1.filterDate(start_date, end_date)
    collectionS1 = collectionS1.select(['VV', 'VH'])
    collectionS1_first_desc = collectionS1.median() 


    # also for acending orbit
    collectionS1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    collectionS1 = collectionS1.filter(ee.Filter.eq('instrumentMode', 'IW'))
    collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    collectionS1 = collectionS1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    collectionS1 = collectionS1.filterBounds(roi)
    # collectionS1 = collectionS1.filter(ee.Filter.contains('.geo', roi))
    collectionS1 = collectionS1.filterDate(start_date, end_date)
    collectionS1 = collectionS1.select(['VV', 'VH'])
    collectionS1_first_asc = collectionS1.median() 

    # fill up the missing data of the descending orbit with the ascending orbit data

    # reference
    # composite_filled = collectionS1_first_desc.unmask(collectionS1_first_asc)



    # non_null_mask = collectionS1_first_desc.mask()

    # # Convert the mask to a feature collection
    # non_null_features = non_null_mask.reduceToVectors(**{
    #     'scale': scale,  # Adjust the scale as necessary
    #     'geometryType': 'polygon',
    #     'geometry': roi,
    #     'eightConnected': False,
    #     'maxPixels': 1e9
    # })

    # # Compute a 1km buffer around the non-null areas
    # buffer_features = non_null_features.map(lambda feature: feature.buffer(1000))

    # # Convert the buffer back to an image
    # buffer_mask = ee.Image().paint(buffer_features, 1).unmask(0)

    # # Update the mask of the descending image
    # desc_with_buffer = collectionS1_first_desc.updateMask(buffer_mask)

    # # Use the unmask function on the descending composite, passing the ascending composite as the argument
    # filled_composite = desc_with_buffer.unmask(collectionS1_first_asc) 



    if url_mode:
        try:
            url = collectionS1_first_desc.getDownloadUrl({
                'scale': scale,
                'format': "GEOTIFF", 
                'region': roi,
                'crs': crs,
                'maxPixels':80000000000,
            })
        except Exception as e:
            print(e)
            print("Error in " + filename + " getting the url, moving on tho the next tile")
            return None
        
        download_tile(url, filename, folder)
        return url

    # Export the image, specifying scale and region.
    task = ee.batch.Export.image.toDrive(
        image = collectionS1_first_desc,
        scale = scale,  
        description = filename, 
        fileFormat="GEOTIFF",  
        folder=folder, 
        region = roi, 
        crs=crs, 
        maxPixels=80000000000,
    )
    start(task)


    filenameacs = filename.split('_')[0] + "Asc_" + filename.split('_')[1]

    # Export the image, specifying scale and region.
    task = ee.batch.Export.image.toDrive(
        image = collectionS1_first_asc,
        scale = scale,  
        description = filenameacs,
        fileFormat="GEOTIFF",  
        folder=folder, 
        region = roi, 
        crs=crs, 
        maxPixels=80000000000,
    )
    start(task)


    return None

# def export_gbuildings(collection_name, confidence_min, bbox, description, folder, scale=10):
def export_gbuildings(roi, filename, folder, confidence_min=0.0, scale=10, crs='EPSG:4326', btype="v3"):
    """
    Function to export a filtered Google Earth Engine collection to Google Drive.

    Args:
    - collection_name (str): name of the GEE collection to filter and export.
    - confidence_min (float): minimum confidence to filter by.
    - roi (list): bounding box to filter by, in the format [minLon, minLat, maxLon, maxLat].
    - filename (str): description for the exported data.
    - folder (str): name of the folder in Google Drive to export the data to.
    - scale (int): resolution of the export in meters (default is 30).
    - crs (str): coordinate reference system of the exported data (default is 'EPSG:4326').

    Returns:
    - None.
    """

    # Load the building footprint dataset
    t = ee.FeatureCollection('GOOGLE/Research/open-buildings/{type}/polygons'.format(type=btype))

    # Apply the confidence filters and clip to the bounding box
    # t_filtered = t.filter(ee.Filter.gte('confidence', confidence_min)).filterBounds(roi)
    t_filtered = t.filterBounds(roi)

    # Define the export parameters
    export_params = {
        'collection': t_filtered,
        'description': filename,
        'folder': folder,
    }

    # Export the data to Google Drive
    task = ee.batch.Export.table.toDrive(**export_params)

    # Start the task
    start(task)


def download(minx, miny, maxx, maxy, name):
    """
    Function to download the data from Google Earth Engine to Drive.
    Inputs:
    - minx, miny, maxx, maxy (float): coordinates of the bounding box.
    - name (str): name of the file to download.
    Returns:
    - None. (the files are downloaded to Drive instead)
    """

    exportarea = { "type": "Polygon",  "coordinates": [[[maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny], [maxx, miny]]]  }
    exportarea = ee.Geometry.Polygon(exportarea["coordinates"]) 

    # transform into local projection
    # find the EPSG code for the local projection
    # exportarea3035 = exportarea.transform('EPSG:3035')

    S1 = True
    # S1 = False

    # S2 = True
    S2 = False

    # S2A = True
    S2A = False

    # VIIRS = True
    VIIRS = False

    # GoogleBuildings = True
    GoogleBuildings = False

    if S1:
        ########################### Processing Sentinel 1 #############################################

        export_S1_tile("spring", (Sen2spring_start_date, Sen2spring_finish_date), "S1spring_" + name, exportarea, name, url_mode=False)
        export_S1_tile("summer", (Sen2summer_start_date, Sen2summer_finish_date), "S1summer_" + name, exportarea, name, url_mode=False)
        export_S1_tile("autumn", (Sen2autumn_start_date, Sen2autumn_finish_date), "S1autumn_" + name, exportarea, name, url_mode=False)
        export_S1_tile("winter", (Sen2winter_start_date, Sen2winter_finish_date), "S1winter_" + name, exportarea, name, url_mode=False)

    if S2:
        ########################### Processing Sentinel 2 Level 1C #############################################
        
        export_cloud_free_sen2("S2spring", (Sen2spring_start_date, Sen2spring_finish_date), name, exportarea, S2type="S2")
        export_cloud_free_sen2("S2summer", (Sen2summer_start_date, Sen2summer_finish_date), name, exportarea)
        export_cloud_free_sen2("S2autumn", (Sen2autumn_start_date, Sen2autumn_finish_date), name, exportarea)
        export_cloud_free_sen2("S2winter", (Sen2winter_start_date, Sen2winter_finish_date), name, exportarea)

    if S2A:
        ########################### Processing Sentinel 2 Level 2A #############################################

        export_cloud_free_sen2("S2Aspring", (Sen2spring_start_date, Sen2spring_finish_date), name, exportarea, S2type="S2_SR_HARMONIZED")
        export_cloud_free_sen2("S2Asummer", (Sen2summer_start_date, Sen2summer_finish_date), name, exportarea, S2type="S2_SR_HARMONIZED")
        export_cloud_free_sen2("S2Aautumn", (Sen2autumn_start_date, Sen2autumn_finish_date), name, exportarea, S2type="S2_SR_HARMONIZED")
        export_cloud_free_sen2("S2Awinter", (Sen2winter_start_date, Sen2winter_finish_date), name, exportarea, S2type="S2_SR_HARMONIZED")
     
    if VIIRS:
        ########################### Processing Sentinel 2 #############################################

        viirs_NL_col = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG') \
                    .filter(ee.Filter.date(Sen2spring_start_date, Sen2winter_finish_date))
        NL_median = viirs_NL_col.select("avg_rad").median()

        # Create composite
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


    if GoogleBuildings:
        ########################### Processing Google Buildings #############################################
        
        # Google Buildings
        export_gbuildings(exportarea, "Gbuildings_" + name, folder=name, confidence_min=0.0, btype="v3") 
        export_gbuildings(exportarea, "Gbuildings_v1_" + name, folder=name, confidence_min=0.0, btype="v1")



    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("minx", type=float)
    parser.add_argument("miny", type=float)
    parser.add_argument("maxx", type=float)
    parser.add_argument("maxy", type=float) 
    parser.add_argument("name", type=str) 
    args = parser.parse_args()

    download(args.minx, args.miny, args.maxx, args.maxy, args.name)


if __name__ == "__main__":
    main()
    print("Done!")


