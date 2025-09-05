import pickle
import os 
from shapely import wkt
import pandas as pd
import geopandas as gpd

sample_dir = '/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/samples/'

for file in [name for name in os.listdir(sample_dir) if 'csv' in name]:
    df = pd.read_csv(sample_dir+f'{file}')
    df.loc[:, 'geometry'] = df.loc[:, 'geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:3857')
    gdf.to_file(sample_dir+f'{file.split(".")[0]}.geojson', driver='GeoJSON', index=False)


with open('/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/database/cache/listdir_lists/listdir_samples.pkl', 'wb') as f:
    pickle.dump([name for name in os.listdir(sample_dir) if 'geojson' in name], f)