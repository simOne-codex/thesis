import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import sys


class PointData():

    def __init__(self):
        self.__data = None

    def get_table(self):
        return self.__data

    def __variable_percentage(self, buffer, clipped, var, vegetazione_gdf):
        possible_values = list(vegetazione_gdf[var].unique())
        areas_ratios = dict.fromkeys(possible_values)
        buffer_area = buffer.area[0]

        for value in areas_ratios.keys():
            piece = clipped[clipped[var] == value]
            areas_ratios[value] = (piece.area.sum()/buffer_area)
    
        return areas_ratios


    def __polygon_coverage_in_buffer(self, gdf, query_point, vars, radius_meters):

        gdf = gdf.to_crs(epsg=3857)  # Web Mercator in meters
        query_point = gpd.GeoSeries([query_point], crs="EPSG:3857").iloc[0]
        # Create buffer around the point
        buffer = gpd.GeoDataFrame([query_point.buffer(radius_meters)], columns=['geometry'], crs='EPSG:3857')
        # Intersect polygon(s) with buffer
        clipped = gpd.overlay(gdf, buffer, how='intersection')
        # Remove empty geometries
        clipped = clipped[~clipped.is_empty]
    
        # Calculate percentage coverage
        result = pd.Series(query_point, index = ['geometry'])

        # Compute ratios for every variable value
        for var in vars:
            areas_ratios = self.__variable_percentage(buffer, clipped, var, gdf)
            for key, ratio in areas_ratios.items():
                name = str(var)+'_'+str(key)
                new_line = pd.Series(ratio, index=[name])
                result = pd.concat([result, new_line], axis=0)

        return result.drop(index='geometry')


    def __missing_georeference(self, df):
        return df.shape[0] == 0


    def gather_point_data(self, per_comune_gdf, altimetria_gdf, vegetazione_gdf, strade_gdf, meteo_gdf, day, query_point, radius_meters=1000):

        # dati per comune
        aux = per_comune_gdf[per_comune_gdf.contains(query_point)]
        if self.__missing_georeference(aux):
            cols = [col for col in per_comune_gdf.columns if col not in ['geometry']]
            data_per_comune = pd.Series([None]*len(cols), index = cols)
        else:
            data_per_comune = pd.Series(aux.loc[aux.index[0], [col for col in aux.columns if col not in ['geometry']]])
        
        # altimetria
        aux = altimetria_gdf[altimetria_gdf.contains(query_point)]
        if self.__missing_georeference(aux):
            cols = [col for col in altimetria_gdf.columns if col not in ['geometry']]
            data_altimetria = pd.Series([None]*len(cols), index = cols)
        else:
            data_altimetria = pd.Series(aux.loc[aux.index[0], [col for col in aux.columns if col not in ['geometry']]])

        # copertura vegetazione nel raggio di radius_meters
        data_vegetazione = self.__polygon_coverage_in_buffer(vegetazione_gdf ,
                                                                       query_point, 
                                                                        [col for col in vegetazione_gdf.columns if col not in ['geometry']],
                                                                        radius_meters)
        # copertura strade nel raggio di radius_meters
        data_strade = self.__polygon_coverage_in_buffer(strade_gdf, 
                                                                query_point, 
                                                                [col for col in strade_gdf.columns if col not in ['geometry']],
                                                                radius_meters)
        # previsioni meteo del punto
        aux = meteo_gdf.merge(gpd.GeoDataFrame({'geometry':[query_point]}, geometry='geometry'), on='geometry', how='inner')
        if self.__missing_georeference(aux):
            cols = [col for col in meteo_gdf.columns if col not in ['geometry', 'day', 'YYYY', 'height']]
            data_meteo = pd.Series([None]*len(cols), index = cols)
        else:
            print('input day:', day)
            print(aux)
            data_meteo = pd.Series(aux[aux.day == day].loc[aux[aux.day == day].index[0],[col for col in aux.columns if col not in ['geometry', 'day', 'YYYY', 'height']]])


        data = pd.concat([data_per_comune, data_altimetria, data_vegetazione, data_strade, data_meteo], axis=0)

        self.__data = pd.DataFrame(data).T.astype('float64')





class PointDataLoader():

    def __init__(self, anno=None, mese=None, giorno=None, x=None, y=None, crs=3857):
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
        
        table = point_data.get_table()
        table['YYYY'] = self.__anno
        table['MM'] = self.__mese
        table['DD'] = self.__giorno
        table['ignition_point (epsg=3857)'] = self.__coordinates

        self.__data = table

    