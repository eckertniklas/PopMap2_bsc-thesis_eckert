# POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2

Detailed population maps play an important role in diverse fields ranging from humanitarian action to urban planning. 
Generating such maps in a timely and scalable manner presents a challenge, especially in data-scarce regions.
To address it we have developed POPCORN, a population mapping method whose only inputs are free, globally available satellite images from Sentinel-1 and Sentinel-2; and a small number of aggregate population counts over coarse census districts for calibration.
Despite the minimal data requirements our approach surpasses the mapping accuracy of existing schemes, including several that rely on building footprints derived from high-resolution imagery.
E.g., we were able to produce population maps for Rwanda with 100$\,$m GSD based on less than 400 regional census counts. 
In Kigali, those maps reach an $R^2$ score of 66% w.r.t. a ground truth reference map, with an average error of only $\pm$10 inhabitants/ha.
Conveniently, POPCORN retrieves explicit maps of built-up areas and of local building occupancy rates, making the mapping process interpretable and offering additional insights, for instance about the distribution of built-up, but unpopulated areas (e.g., industrial warehouses).
Moreover, we find that, once trained, the model can be applied repeatedly to track population changes; and that it can be transferred to geographically similar regions with only a moderate loss in performance (e.g., from Uganda to Rwanda).
With our work we aim to democratize access to up-to-date and high-resolution population maps, recognizing that some regions faced with particularly strong population dynamics may lack the resources for costly micro-census campaigns.

![Bunia Time Series](imgs/series_bunia.jpg)

## Methodology

![Graphical Abstract](imgs/graphical_abstract_v17.jpg)

## Installation 

Instructions on how to install the project or library.

Set up the base environment like this:
```bash
python -m venv PopMapEnv
source PopMapEnv/bin/activate
pip install requirements.txt
```
Additionally, install GDAL without sudo access  as described in this [post](https://askubuntu.com/questions/689065/how-could-i-install-gdal-without-root)
 - download the gdal-3.4.1 binary, and extract it.
 - execute the commands (this might take some time):
```bash
./autogen.sh
./configure
make
```




