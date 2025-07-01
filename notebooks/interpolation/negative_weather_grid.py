import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *
import weather_forecast_utils
reload(weather_forecast_utils)
from weather_forecast_utils import *

from shapely.ops import unary_union
from shapely.geometry import Point
import random
random.seed(42569)

import pickle 
with open('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/data_for_weather_kriging.pkl', 'rb') as f:
     weather_dict = pickle.load(f)
aux = separate_date(gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/nicola/piedmont_2012_2024_fa.geojson'), 'initialdate')
ignitions = aux[aux['YYYY'] >= 2016]
confini = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/clean_data/confini_piemonte/confini_piemonte.shp').to_crs(epsg=3857)


def month_day_to_day_number(year, month, day):
    feb = 28
    if year in list(range(2000, 2025, 4)):
        feb += 1
            
    days_per_month = [31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_of_year = sum(days_per_month[:month - 1]) + day
    return day_of_year
ignitions['day'] = ignitions.apply(lambda x: month_day_to_day_number(x['YYYY'], x['MM'], x['DD']), axis=1)



fires = pd.concat([ignitions[['YYYY', 'day', 'MM', 'DD']].reset_index(drop=True), pd.Series([Point(x, y) for x, y in zip(ignitions.point_x, ignitions.point_y)])], axis=1).rename(columns={0: 'geometry'})


def create_random_point(min_year, max_year, fires, polygons, buffer_distance = 1000):
    year = random.randint(min_year, max_year)
    total = 365
    if year in list(range(2000, 2025, 4)):
        total += 1
    day = random.randint(1, total)

    daily_fires = fires[(fires['YYYY'] == year)]
    daily_fires = daily_fires[daily_fires['day'] == day]
    if not daily_fires.shape[0] == 0:
        buffers = gpd.GeoDataFrame(gpd.GeoDataFrame(daily_fires, geometry='geometry', crs="EPSG:3857").geometry.buffer(buffer_distance), 
                                   columns=['geometry']).union_all(method='unary')
        geometry_check = polygons.union_all(method='unary').difference(buffers)
    else:
        geometry_check = polygons.union_all(method='unary')
    
    pminx, pminy, pmaxx, pmaxy = geometry_check.bounds

    random_point = Point(random.uniform(pminx, pmaxx), random.uniform(pminy, pmaxy))
    while not geometry_check.contains(random_point):
        random_point = Point(random.uniform(pminx, pmaxx), random.uniform(pminy, pmaxy))

    return gpd.GeoDataFrame({'YYYY': {0:year}, 'day':{0:day}, 'geometry': {0:random_point}}, crs="EPSG:3857")


n_points = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/table_data_input.csv').shape[0]
negative_grid = gpd.GeoDataFrame()

for _ in tqdm(range(n_points)):
    negative_grid = pd.concat([negative_grid, create_random_point(2016, 2024, fires, confini)],
                               ignore_index=True)


save_clean_data(negative_grid, 'negative_fires_grid', '/nfs/home/genovese/thesis-wildfire-genovese/data/gathering_geojson')