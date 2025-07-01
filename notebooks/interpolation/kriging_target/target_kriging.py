import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *

positives = gpd.GeoDataFrame(
    pd.DataFrame(gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/kriging_weather_grid.geojson')),
    ).set_crs(epsg=3857, allow_override=True)
negatives = gpd.GeoDataFrame(pd.DataFrame(
    gpd.read_file('/nfs/home/genovese/thesis-wildfire-genovese/data/data_loader_for_kriging/kriging_negative_weather_grid.geojson')),
    crs='EPSG:3857')

p = pd.concat([positives[['geometry']], pd.Series([1]*positives.shape[0], name='label')],  axis=1)
n = pd.concat([negatives[['geometry']], pd.Series([0]*negatives.shape[0], name='label')],  axis=1)

tot = pd.concat([p, n])


import numpy as np
from pykrige.rk import Krige
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Define parameter grid for Kriging
param_grid = {
    "method": ["ordinary"],
    "variogram_model": ["linear", "power", "exponential", "gaussian", "spherical"],
}

# Create Kriging model (scikit-learn wrapper)
kriging_model = Krige()

# GridSearch with 10-fold cross-validation
grid_search = GridSearchCV(
    estimator=kriging_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=10,
    verbose=10,
    n_jobs=-1
)

# Fit GridSearch
print('Gets here')
grid_search.fit(np.array(pd.concat([tot.geometry.x, tot.geometry.y], axis=1)), tot.label)
print('passes over')

result = pd.DataFrame(grid_search.cv_results_)
print('time to save')
save_clean_data(result, f"ordinary_target_kriging", '/nfs/home/genovese/thesis-wildfire-genovese/outputs/grid_searches')

# Print best model
print("Best parameters:", grid_search.best_params_)
print("Best MSE:", -grid_search.best_score_)
