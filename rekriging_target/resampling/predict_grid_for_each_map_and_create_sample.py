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

grid = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/outputs/kriged_map/grid_100m_piedmont.geojson')


file_dir = '/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/kriged_maps/'

for map in tqdm(os.listdir(file_dir), desc='Predicting map values...'):
    with open(file_dir+f'{map}', 'rb') as f:
        model = pickle.load(f)
    values = model.predict(np.array(pd.concat([grid.geometry.x, grid.geometry.y], axis=1)))
    pd.Series(values, name='target').to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/kriged_map_values/{map.split(".")[0]}.csv', index=False)


val_dir = '/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/kriged_map_values/'

for values in tqdm(os.listdir(val_dir), desc='Sampling per map...'):
    aux = pd.concat([grid, pd.read_csv(val_dir+f'{values}')], axis=1)
    aux.loc[: ,'target'] = aux.loc[:, 'target'].round(1)

    sample = gpd.GeoDataFrame(columns=['geometry', 'target'])

    for value, gdf in aux.groupby('target'):
        foo = gdf.sample(1000, random_state=random_state+int(value*25))
        sample = pd.concat([sample, foo], axis=0, ignore_index=True)
    sample.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/samples/{values}', index=False)




with open('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/data_for_weather_kriging.pkl', 'rb') as f:
     loaded_dict = pickle.load(f)

for year in range(2000, 2016):
     daily_forecast(loaded_dict, year, [list(loaded_dict[year].keys())[0]])



sample_dir = '/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/samples/'

is_beginning = True

for sample in tqdm(os.listdir(sample_dir), desc='Loading data for each sample...'):
    
    final_data = pd.DataFrame()

    fires = pd.read_csv(sample_dir+f'{sample}')
    crs = fires.crs

    for f in tqdm(fires.index, desc=f'Loading data for sample {sample}'):
        row = fires.loc[f, :]
        data = PointDataLoader(anno = sample.split("_")[0],
                               stagione = sample.split("_")[1].split(".")[0],
                                x = row.geometry.x,
                                y = row.geometry.y,
                                crs = crs)
        data.load()
        table = data.get_table()
        final_data = pd.concat([final_data, table], ignore_index=True)

    if is_beginning:
        final_data.to_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/table_data_input.csv', index=False)
        is_beginning=False
    else:
        current = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/table_data_input.csv')
        pd.concat([current, final_data], ignore_index=True
              ).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/model_input/table_data_input.csv', index=False)


