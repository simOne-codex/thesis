import pickle
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import geopandas as gpd
import os

sys.path.append('/nfs/home/genovese/thesis-wildfire-genovese/src/')
import utils
from importlib import reload
reload(utils)
from utils import *
import weather_forecast_utils
reload(weather_forecast_utils)
from weather_forecast_utils import *
sys.path.append('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/')
import PointDataLoader
reload(PointDataLoader)
from PointDataLoader import *

with open('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/data_for_weather_kriging.pkl', 'rb') as f:
     loaded_dict = pickle.load(f)

year = 2000
daily_forecast(loaded_dict, year, list(loaded_dict[year].keys()))
