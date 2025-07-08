import os
import sys
# sys.path.append(os.path.abspath(rf"/nfs/home/genovese/thesis-wildfire-genovese/src/OnFireForest/"))
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src/OnFireForest/")
import random_forest_only
from importlib import reload
reload(random_forest_only)
from random_forest_only import run_rf_only

run_rf_only(
    input_csv_path='/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/not_imputed_dataset_non_kriged.csv',
    regression=False,
    results_file_name='classification_non_kriged',
    random_state = 89
    )