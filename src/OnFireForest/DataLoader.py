import os
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm


class DataLoader():

    def __init__(self, anno=None, mese=None, giorno=None, xmin=None, ymin=None, xmax=None, ymax=None):
        self.__anno = anno
        self.__mese = mese
        self.__giorno = giorno
        self.total_bounds = (xmin, ymin, xmax, ymax)
        self.__data = pd.DataFrame()
        if (pd.Series([anno, mese, giorno, xmin, xmax, ymin, ymax]).isna().any()):
            self.initialised_ = False
        else:
            self.initialised_ = True
        
    
    def get_table(self):
        return self.__data
    
    def get_coordinates(self, params=None):
        result = {'anno': self.__anno,
                  'mese': self.__mese,
                  'giorno': self.__giorno,
                  'coordinate': self.total_bounds}
        if params is None:
            return result
        else:
            return result[params]


    def set_coordinates(self, anno, mese, giorno, xmin, ymin, xmax, ymax):
        self.anno_ = anno
        self.mese_ = mese
        self.giorno_ = giorno
        self.total_bounds = (xmin, ymin, xmax, ymax)
        self.initialised_ = True


        
    def load(self):
    
        # if not self.initialised_:
        #     raise Exception('Space and time coordinates must be set (use method self.set_coordinates)')
        # db = '/nfs/home/genovese/thesis-wildfire-genovese/database/'
    
        pass

    