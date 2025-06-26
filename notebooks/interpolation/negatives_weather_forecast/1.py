import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *
import weather_forecast_utils
reload(weather_forecast_utils)
from weather_forecast_utils import *

import pickle 
with open('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/data_for_weather_kriging.pkl', 'rb') as f:
     loaded_dict = pickle.load(f)

points = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/negative_fires_grid.geojson')

for y in tqdm(list(points.YYYY.unique())):
     save_clean_data(yearly_forecast(loaded_dict, points, y), str(y), '/nfs/home/genovese/thesis-wildfire-genovese/data/negative_output_weather_kriging')