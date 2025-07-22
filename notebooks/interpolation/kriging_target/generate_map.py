import pickle 
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os
from tqdm import tqdm

with open('/nfs/home/genovese/thesis-wildfire-genovese/outputs/kriged_map/target_kriging.pkl', 'rb') as f:
    krige = pickle.load(f)

if not os.path.exists('/nfs/home/genovese/thesis-wildfire-genovese/outputs/kriged_map/grid_100m_piedmont.geojson'):
    confini = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/clean_data/confini_piemonte/confini_piemonte.shp').to_crs(epsg=3857)
    polygon = confini.union_all(method='unary')

    spacing = 100  # meters

    # Get bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    x_coords = np.arange(minx, maxx + spacing, spacing)
    y_coords = np.arange(miny, maxy + spacing, spacing)

    points = [Point(x, y) for x in tqdm(x_coords, desc='Run raw point grid...') for y in y_coords]

    inside_points = [pt for pt in tqdm(points, 'Selecting only inside points...') if polygon.contains(pt)]

    points_gdf = gpd.GeoDataFrame(inside_points, columns=['geometry']).set_crs('epsg:3857')
    points_gdf.to_file('/nfs/home/genovese/thesis-wildfire-genovese/database/grid_100m_piedmont.geojson', driver='GeoJSON', index=False)

grid = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/outputs/kriged_map/grid_100m_piedmont.geojson')
predictions = krige.predict(np.array(pd.concat([grid.geometry.x, grid.geometry.y], axis=1)))

with open('/nfs/home/genovese/thesis-wildfire-genovese/outputs/kriged_map/distribution_kriged_map.pkl', 'wb') as f:
    pickle.dump(predictions, f)