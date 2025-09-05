from importlib import reload
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import sys
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/resampling/src/")
import PointDataLoader
reload(PointDataLoader)
from PointDataLoader import PointDataLoader

fires = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/database/cache_resampled/dataset.geojson')

final_data = pd.DataFrame()
crs = fires.crs

for f in tqdm(fires.index[10340:10410]):
    row = fires.loc[f, :]
    data = PointDataLoader(anno = row.YYYY,
                           x = row.geometry.x,
                           y = row.geometry.y,
                           crs = crs)
    data.load()
    table = data.get_table()
    final_data = pd.concat([final_data, table], ignore_index=True)

current = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input_resampled/table_data_input.csv')
pd.concat([current, final_data], ignore_index=True).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input_resampled/table_data_input.csv', index=False)