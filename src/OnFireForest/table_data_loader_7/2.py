import os
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import sys
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src/OnFireForest/")
from PointDataLoader import PointDataLoader

fires = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/kriging_ambiguous_weather_grid.geojson')

final_data = pd.DataFrame()
crs = fires.crs

for f in tqdm(fires.index[1280:1360]):
    row = fires.loc[f, :]
    data = PointDataLoader(anno = row.YYYY,
                           mese = row.MM,
                           giorno = row.DD,
                           x = row.geometry.x,
                           y = row.geometry.y,
                           crs = crs)
    data.load()
    table = data.get_table()
    final_data = pd.concat([final_data, table], ignore_index=True)


current = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/ambiguous_table_data_input.csv')
pd.concat([current, final_data], ignore_index=True).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/ambiguous_table_data_input.csv', index=False)