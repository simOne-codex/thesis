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
random.seed(56161)

confini = gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/clean_data/confini_piemonte/confini_piemonte.shp').to_crs(epsg=3857)

main_dir = '/nfs/home/genovese/thesis-wildfire-genovese/rekriging_target/'
positive_grid = gpd.read_file(main_dir+'database/cache/positive_grid.geojson')


from shapely.geometry import Point
import random
import geopandas as gpd
from tqdm import tqdm


def create_random_point(year, min_month, max_month, fires, polygons, buffer_distance = 1000):
    month = random.randint(min_month, max_month)

    if not fires.shape[0] == 0:
        buffers = gpd.GeoDataFrame(gpd.GeoDataFrame(fires, geometry='geometry', crs="EPSG:3857").geometry.buffer(buffer_distance), 
                                   columns=['geometry']).union_all(method='unary')
        geometry_check = polygons.union_all(method='unary').difference(buffers)
    else:
        geometry_check = polygons.union_all(method='unary')
    
    pminx, pminy, pmaxx, pmaxy = geometry_check.bounds

    random_point = Point(random.uniform(pminx, pmaxx), random.uniform(pminy, pmaxy))
    while not geometry_check.contains(random_point):
        random_point = Point(random.uniform(pminx, pmaxx), random.uniform(pminy, pmaxy))

    return gpd.GeoDataFrame({'YYYY': {0:year}, 'MM': {0:month}, 'geometry': {0:random_point}}, crs="EPSG:3857")



for y, year_df in tqdm(positive_grid.groupby('YYYY')):
    winter = year_df[year_df.MM.isin([1,2,3])]
    spring = year_df[year_df.MM.isin([4,5,6])]
    summer = year_df[year_df.MM.isin([7,8,9])]
    autumn = year_df[year_df.MM.isin([10,11,12])]
    for season, season_df in tqdm(zip(['winter', 'spring', 'summer', 'autumn'], [winter, spring, summer, autumn])):
        n_points = season_df.shape[0]
        negative_grid = gpd.GeoDataFrame()
        for _ in tqdm(range(n_points)):
            negative_grid = pd.concat([negative_grid, create_random_point(y, 
                                                                        season_df.MM.unique().min(), 
                                                                        season_df.MM.unique().max(), 
                                                                        positive_grid, 
                                                                        confini)], 
                                                                        ignore_index=True)

        negative_grid.to_csv(main_dir+f'database/cache/by_season/{y}_{season}_negative.csv')
        season_df.to_csv(main_dir+f'database/cache/by_season/{y}_{season}_positive.csv')
