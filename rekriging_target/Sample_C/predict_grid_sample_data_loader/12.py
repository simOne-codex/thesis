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


random_state = 9911

grid = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/grid_1000m_piedmont.geojson')


file_dir = '/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/kriged_maps/'


with open('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/listdir_lists/grid_array.pkl', 'rb') as s:
    grid_array = pickle.load(s)

with open('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/listdir_lists/listdir_kriged_maps.pkl', 'rb') as f:
    listdir_kriged_maps = pickle.load(f)


random_state = 105

for map in tqdm(listdir_kriged_maps[99:], desc='Predicting map values...'):
    with open(file_dir+f'{map}', 'rb') as f:
        model = pickle.load(f)
    values = model.predict(grid_array)
    value_series = pd.Series(values, name='target')
    
    value_series.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/kriged_map_values/{map.split(".")[0]}.csv', index=False)

    aux = pd.concat([grid, value_series], axis=1)
    aux.loc[: ,'target'] = aux.loc[:, 'target'].round(1)
    n_minor_class = aux.target.value_counts().iloc[-1]
    sample = gpd.GeoDataFrame(columns=['geometry', 'target'])

    for value, gdf in aux.groupby('target'):
        foo = gdf.sample(np.min([n_minor_class, 50]), random_state=random_state+int(value*25))
        sample = pd.concat([sample, foo], axis=0, ignore_index=True)
    sample.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/samples/{map.split(".")[0]}.csv', index=False)