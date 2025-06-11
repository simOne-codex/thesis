import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *
import weather_forecast_utils
reload(weather_forecast_utils)
from weather_forecast_utils import *
data_folder = "/nfs/home/genovese/thesis-wildfire-genovese/data/"

import pickle 
with open('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/data_for_weather_kriging.pkl', 'rb') as f:
     loaded_dict = pickle.load(f)
     
print(list(loaded_dict.keys()))

ignitions = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/kriging_weather_grid.geojson'
                       ).rename(columns={'day_of_year': 'day'}) 

save_clean_data(yearly_forecast(loaded_dict, ignitions, 2019), str(2019), '/nfs/home/genovese/thesis-wildfire-genovese/data/output_weather_kriging')