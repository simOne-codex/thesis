import sys
sys.path.append(rf'/nfs/home/genovese/thesis-wildfire-genovese/src/')
from importlib import reload
import utils
reload(utils)
from utils import *
data_folder = '/nfs/home/genovese/thesis-wildfire-genovese/data/'

target = separate_date(gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/nicola/piedmont_2012_2024_fa.geojson'), 'initialdate')
def month_day_to_day_number(year, month, day):
    feb = 28
    if year in list(range(2000, 2025, 4)):
        feb += 1
            
    days_per_month = [31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_of_year = sum(days_per_month[:month - 1]) + day
    return day_of_year
target['day_of_year'] = target.apply(lambda x: month_day_to_day_number(x['YYYY'], x['MM'], x['DD']), axis=1)
altimetria = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson/altimetria_per_circoscrizione.geojson').to_crs(target.crs)


from shapely.geometry import Point

target['ignition_point'] = [Point(x, y) for x, y in tqdm(zip(target.point_x, target.point_y))]

def get_polygon_value(gdf, point, value):
    for idx, polygon in gdf.iterrows():
        if polygon['geometry'].contains(point):
            return polygon[value]
    return None

# Apply the function to all points
values = [get_polygon_value(altimetria, point, 'MEDIANA') for point in tqdm(target['ignition_point'])]

# 5. Create a GeoDataFrame with points and associated polygon values
points_gdf = gpd.GeoDataFrame(
    geometry=target['ignition_point'],
    data={'height': values},
    crs=altimetria.crs
).rename(columns={'ignition_point': 'geometry'})

for_kriging = gpd.GeoDataFrame(target[['day_of_year', 'ignition_point', 'YYYY']], geometry='ignition_point'
                               ).rename(columns={'ignition_point': 'geometry'}).merge(points_gdf, on='geometry', how='inner')

for_kriging.to_file('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/kriging_weatehr_grid.geojson')