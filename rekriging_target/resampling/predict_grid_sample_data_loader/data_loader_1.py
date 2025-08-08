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


sample_dir = '/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/samples/'

with open('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/listdir_lists/listdir_samples.pkl', 'rb') as f:
    listdir_samples = pickle.load(f)

is_beginning = True

for sample in tqdm(listdir_samples[:9], desc='Loading data for each sample...'):
    
    final_data = pd.DataFrame()

    fires = gpd.read_file(sample_dir+f'{sample}')
    crs = fires.crs

    for f in tqdm(fires.index, desc=f'Loading data for sample {sample}'):
        row = fires.loc[f, :]
        data = PointDataLoader(anno = sample.split("_")[0],
                               stagione = sample.split("_")[1].split(".")[0],
                                x = row.geometry.x,
                                y = row.geometry.y,
                                target = row['target'],
                                crs = crs)

        data.load()
        table = data.get_table()
        final_data = pd.concat([final_data, table], ignore_index=True)

    current = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/table_data_input.csv')
    pd.concat([current, final_data], ignore_index=True
              ).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/table_data_input.csv', index=False)




