import sys
sys.path.append(rf'/nfs/home/genovese/thesis-wildfire-genovese/src/')
from importlib import reload
import utils
reload(utils)
from utils import *


from sklearn.ensemble import GradientBoostingRegressor
from pykrige.rk import RegressionKriging
import pickle

def rk_train_pred(X_train, X_pred, target_col):
    
    # no data case
    if X_train[target_col].isna().all(): 
        return [None]*X_train.shape[0]
    
    # univariate case
    elif X_train[target_col].nunique() == 1: 
        y_pred = [X_train[target_col].iloc[0]]*X_train.shape[0]
    
    # interpolable case
    else: 
        RK = RegressionKriging(regression_model = GradientBoostingRegressor(), method='universal', variogram_model = 'spherical', verbose=False)
       
        RK.fit(p = X_train[['height']], x = np.transpose(np.array([X_train.geometry.x, X_train.geometry.y])), y = X_train[target_col])
       
        y_pred = RK.predict(p = X_pred[['height']], x = np.transpose(np.array([X_pred.geometry.x, X_pred.geometry.y])))

    return y_pred



def yearly_forecast(loaded_dict, target, year):
    
    centraline = loaded_dict[year]
    tt = target[target['YYYY'] == year]

    forecasts = tt.copy()
    
    for var, gdf in tqdm(centraline.items(), desc=f'Running year: {year}'):

        parameter_forecast = gpd.GeoDataFrame()

        for coord, X_pred in tqdm(tt.groupby(['day', 'geometry']), desc=f'Running variable: {var}'):
            aux = X_pred.copy().reset_index(drop=True)
            X_train = gdf[gdf.day == coord[0]].to_crs(epsg=3857)
            y_pred = rk_train_pred(X_train, X_pred, var)
            aux = pd.concat([aux, pd.DataFrame(pd.Series(y_pred), columns=[var])], axis=1)
            parameter_forecast = pd.concat([parameter_forecast, aux], axis=0, ignore_index=True)

        forecasts = forecasts.merge(parameter_forecast[['geometry', var]], on='geometry', how='inner')

    return forecasts




class ConstantPredictor():
    def __init__(self, constant_value):
        self.constant_value = constant_value

    def predict(self, input_array):
        return np.array(self.constant_value, input_array.shape[1])





def rk_train(X_train, target_col):

    # no data case
    if X_train[target_col].isna().all(): 
        return ConstantPredictor(None)
    
    # univariate case
    elif X_train[target_col].nunique() == 1: 
        return ConstantPredictor(X_train[target_col].unique()[0])
    
    # interpolable case
    else: 
        RK = RegressionKriging(regression_model = GradientBoostingRegressor(), method='universal', variogram_model = 'spherical', verbose=False)
        RK.fit(p = X_train[['height']], x = np.transpose(np.array([X_train.geometry.x, X_train.geometry.y])), y = X_train[target_col])
        return RK




def daily_forecast(loaded_dict: dict, year: int, vars: list):
    centraline = loaded_dict[year]
    
    for var in tqdm(vars, desc=f'Running year: {year}'):
        gdf =  centraline[var]
        
        for day in tqdm(list(gdf.day.unique()), desc=f'Running variable: {var}'):
            X_train = gdf[gdf.day == day].to_crs(epsg=3857)
            model = rk_train(X_train, var)
            with open(f'/nfs/home/genovese/thesis-wildfire-genovese/database/daily_weather_maps/{year}_{day}_{var}.pkl', 'wb') as f:
                pickle.dump(model, f)

    return