import os
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point
from importlib import reload
import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src/OnFireForest")
import PointData
reload(PointData)
from PointData import PointData


class PointDataLoader():

    def __init__(self, anno=None, mese=None, giorno=None, x=None, y=None, crs=None):
        self.__anno = anno
        self.__mese = mese
        self.__giorno = giorno
        self.__data = pd.DataFrame()

        if (x is None) | (y is None):
            self.__coordinates = None
        else:
            self.__coordinates = gpd.GeoSeries(Point(x, y), crs=crs).to_crs(epsg=3857).iloc[0]

        if (pd.Series([anno, mese, giorno, x, y]).isna().any()):
            self.initialised_ = False
        else:
            self.initialised_ = True 
        
    
    def get_table(self):
        return self.__data
    
    def get_coordinates(self, params=None):
        result = {'anno': self.__anno,
                  'mese': self.__mese,
                  'giorno': self.__giorno,
                  'coordinate': self.__coordinates}
        return result


    def set_coordinates(self, anno, mese, giorno, x, y, crs):
        self.__anno = anno
        self.__mese = mese
        self.__giorno = giorno
        self.__coordinates = gpd.GeoSeries(Point(x, y), crs=crs).to_crs(epsg=3857).iloc[0]
        self.initialised_ = True


    def __month_day_to_day_number(self, year, month, day):
        feb = 28
        if year in list(range(2000, 2025, 4)):
            feb += 1
            
        days_per_month = [31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_of_year = sum(days_per_month[:month - 1]) + day
        return day_of_year
    


    def load(self, db_path = '/nfs/home/genovese/thesis-wildfire-genovese/database/'):
        if not self.initialised_:
            raise Exception('Space and time coordinates must be set (use method self.set_coordinates)')

        if self.__data.shape[0] > 0:
            print('Data already loaded')
            return

        point_data = PointData()

        per_comune_path = os.path.join(db_path, f'per_comune_{self.__anno}.geojson')
        altimetria_path = os.path.join(db_path, f'altimetria.geojson')
        vegetazione_path = os.path.join(db_path, f'vegetazione.geojson')
        strade_path = os.path.join(db_path, f'strade.geojson')
        meteo_path = os.path.join(db_path, f'meteo_{self.__anno}.geojson')

        point_data.gather_point_data(gpd.read_file(per_comune_path).to_crs(epsg=3857),
                                     gpd.read_file(altimetria_path).to_crs(epsg=3857),
                                     gpd.read_file(vegetazione_path).to_crs(epsg=3857),
                                     gpd.read_file(strade_path).to_crs(epsg=3857),
                                     gpd.read_file(meteo_path),
                                     day = self.__month_day_to_day_number(self.__anno, self.__mese, self.__giorno),
                                     query_point=self.__coordinates,
                                     radius_meters=1000)
        
        self.__data = point_data.get_table()

    