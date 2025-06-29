import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *
import weather_forecast_utils
reload(weather_forecast_utils)
from weather_forecast_utils import *


from shapely.geometry import Point

def month_day_to_day_number(year, month, day):
    feb = 28
    if year in list(range(2000, 2025, 4)):
        feb += 1
            
    days_per_month = [31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_of_year = sum(days_per_month[:month - 1]) + day
    return day_of_year

target = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/negative_fires_grid.geojson')
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

for_kriging = gpd.GeoDataFrame(target[['YYYY', 'day', 'geometry']], geometry='geometry'
                               ).merge(points_gdf, on='geometry', how='inner')

save_clean_data(for_kriging.dropna(), 'kriging_negative_weather_grid', '/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/')