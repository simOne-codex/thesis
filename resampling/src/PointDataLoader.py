import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np
import pickle

class PointData():

    def __init__(self, year):
        self.__data = None
        self.__year = year

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


    def __identify_month(self, month, year):
        feb=28
        if year in list(range(2000, 2025, 4)):
            feb += 1       

        transform = {1:     [1,                                     1+31], 
                     2:     [1+31,                                  1+31+feb], 
                     3:     [1+31+feb,                              1+31+feb+31], 
                     4:     [1+31+feb+31,                           1+31+feb+31+30], 
                     5:     [1+31+feb+31+30,                        1+31+feb+31+30+31], 
                     6:     [1+31+feb+31+30+31,                     1+31+feb+31+30+31+30], 
                     7:     [1+31+feb+31+30+31+30,                  1+31+feb+31+30+31+30+31], 
                     8:     [1+31+feb+31+30+31+30+31,               1+31+feb+31+30+31+30+31+31], 
                     9:     [1+31+feb+31+30+31+30+31+31,            1+31+feb+31+30+31+30+31+31+30], 
                     10:    [1+31+feb+31+30+31+30+31+31+30,         1+31+feb+31+30+31+30+31+31+30+31], 
                     11:    [1+31+feb+31+30+31+30+31+31+30+31,      1+31+feb+31+30+31+30+31+31+30+31+30], 
                     12:    [1+31+feb+31+30+31+30+31+31+30+31+30,   1+31+feb+31+30+31+30+31+31+30+31+30+31]}
        
        return transform[month] 


    def gather_point_data(self, per_comune_gdf, altimetria_gdf, vegetazione_gdf, strade_gdf, variabili_meteo, query_point, radius_meters=1000):

        # dati per comune
        aux = per_comune_gdf[per_comune_gdf.contains(query_point)]
        if self.__missing_georeference(aux):
            cols = [col for col in per_comune_gdf.columns if col not in ['geometry']]
            data_per_comune = pd.Series([None]*len(cols), index = cols)
        else:
            data_per_comune = pd.Series(aux.loc[aux.index[0], [col for col in aux.columns if col not in ['geometry']]])
        
        # # altimetria DON'T INCLUDE IN FINAL DATA LEAKAGE
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
        """
        lst = []
        for month in range(1, 13):
            lst.extend([f"{param}_{month}" for param in variabili_meteo])
        data_meteo = pd.DataFrame({key: {0: None} for key in lst})
        

        height = np.array(data_altimetria[['ALTIMETRIA_MEDIANA']][0]).reshape(1, -1)
        point = np.array([query_point.x,query_point.y]).reshape(1,2)
        
        for var, month in zip(variabili_meteo, range(1, 13)):
            foo = self.__identify_month(month=month, year=self.__year)
            krige_pred = []
            for dd in range(foo[0], foo[1]):
                with open(f'/nfs/home/genovese/thesis-wildfire-genovese/database/daily_weather_maps/{self.__year}_{dd}_{var}.pkl', 'rb') as f:
                    model = pickle.load(f)
                krige_pred.append(model.predict(p = height, x = point))

            data_meteo.loc[0, f"{var}_{month}"] = np.mean(krige_pred)

        data_meteo = pd.Series(data_meteo.iloc[0, :])
        """
        data = pd.concat([data_per_comune, data_vegetazione, data_strade], axis=0) # data meteo missing

        self.__data = pd.DataFrame(data).T.astype('float64')





class PointDataLoader():

    def __init__(self, anno=None, x=None, y=None, crs=3857):
        self.__anno = anno
        self.__data = pd.DataFrame()

        if (x is None) | (y is None):
            self.__coordinates = None
        else:
            self.__coordinates = gpd.GeoSeries(Point(x, y), crs=crs).to_crs(epsg=3857).iloc[0]

        if (pd.Series([anno, x, y]).isna().any()):
            self.initialised_ = False
        else:
            self.initialised_ = True 
        
    
    def get_table(self):
        return self.__data
    
    def get_coordinates(self):
        result = {'anno': self.__anno,
                  'coordinate': self.__coordinates}
        return result


    def set_coordinates(self, anno, x, y, crs):
        self.__anno = anno
        self.__coordinates = gpd.GeoSeries(Point(x, y), crs=crs).to_crs(epsg=3857).iloc[0]
        self.initialised_ = True    


    def load(self, db_path = '/nfs/home/genovese/thesis-wildfire-genovese/database/'):
        if not self.initialised_:
            raise Exception('Space and time coordinates must be set (use method self.set_coordinates)')

        if self.__data.shape[0] > 0:
            print('Data already loaded')
            return

        point_data = PointData(year=self.__anno)

        per_comune_path = os.path.join(db_path, f'per_comune_{self.__anno}.geojson')
        altimetria_path = os.path.join(db_path, f'altimetria.geojson')
        vegetazione_path = os.path.join(db_path, f'vegetazione.geojson')
        strade_path = os.path.join(db_path, f'strade.geojson')
        meteo_path = os.path.join(db_path, f'meteo_{self.__anno}.geojson')

        point_data.gather_point_data(per_comune_gdf=gpd.read_file(per_comune_path).to_crs(epsg=3857),
                                     altimetria_gdf=gpd.read_file(altimetria_path).to_crs(epsg=3857),
                                     vegetazione_gdf=gpd.read_file(vegetazione_path).to_crs(epsg=3857),
                                     strade_gdf=gpd.read_file(strade_path).to_crs(epsg=3857),
                                     variabili_meteo=['tmedia', 'tmax', 'tmin', 
                                                      'ptot', 'vmedia', 'vraffica', 
                                                      'settore_prevalente', 'tempo_permanenza', 'durata_calma', 
                                                      'umedia', 'umin', 'umax', 
                                                      'rtot', 'hdd_base18', 'hdd_base20', 
                                                      'cdd_base18'],
                                     query_point=self.__coordinates)
        
        table = point_data.get_table()
        table['YYYY'] = self.__anno
        table['ignition_point (epsg=3857)'] = self.__coordinates

        self.__data = table

    