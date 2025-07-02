import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *
import weather_forecast_utils
reload(weather_forecast_utils)
from weather_forecast_utils import *

target = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/ambiguous_fires_grid.geojson')
altimetria = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/altimetria_per_circoscrizione.geojson').to_crs(target.crs)

def get_polygon_value(gdf, point, value):
    for idx, polygon in gdf.iterrows():
        if polygon['geometry'].contains(point):
            return polygon[value]
    return None

# Apply the function to all points
values = [get_polygon_value(altimetria, point, 'MEDIANA') for point in tqdm(target['geometry'])]

# 5. Create a GeoDataFrame with points and associated polygon values
points_gdf = gpd.GeoDataFrame(
    data={'geometry': target['geometry'], 'height': values},
    geometry='geometry',
    crs=altimetria.crs
)

for_kriging = gpd.GeoDataFrame(target[['YYYY', 'day', 'MM', 'DD', 'geometry']], geometry='geometry'
                               ).merge(points_gdf, on='geometry', how='inner')

save_clean_data(for_kriging.dropna(), 'kriging_ambiguous_weather_grid', '/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/')