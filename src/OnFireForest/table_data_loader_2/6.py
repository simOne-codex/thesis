import os
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import sys
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src/OnFireForest/")
from PointDataLoader import PointDataLoader

fires = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/database/piedmont_2016_2024_fa.geojson')

final_data = pd.DataFrame()
crs = fires.crs

for f in tqdm(fires.index[1040:1100]):
    row = fires.loc[f, :]
    data = PointDataLoader(anno = row.YYYY,
                           mese = row.MM,
                           giorno = row.DD,
                           x = row.point_x,
                           y = row.point_y,
                           crs = crs)
    data.load()
    table = data.get_table()
    table['fire_id'] = row['id']
    final_data = pd.concat([final_data, table], ignore_index=True)


current = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/table_data_input.csv')
pd.concat([current, final_data], ignore_index=True).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/table_data_input.csv', index=False)