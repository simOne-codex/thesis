import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, recall_score, f1_score
from tqdm import tqdm
import inspect

import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
import processer
reload(utils)
reload(processer)
from utils import *
from processer import preprocess


def run_rf_only(input_csv_path: str, regression: bool, results_file_name: str, random_state: int = 3469):

    df = pd.read_csv(input_csv_path).drop_duplicates()

    if regression:
        y = 'target'
    else:
        y= 'label'

    non_train_columns = ['target', 'geometry', 'YYYY', 'DD', 'MM', 'day', 'id', 'fire_id', 'label']
    target_column = y
    predictor_columns = [col for col in df.columns if col not in non_train_columns]

    test_df = df.sample(frac=0.15, random_state = random_state + 185)
    test_indices = test_df.index
    X_test = test_df.loc[test_indices, predictor_columns]
    y_test = test_df.loc[test_indices, target_column]

    train_indices = [idx for idx in df.index if idx not in test_indices]
    train_df = df.loc[train_indices, predictor_columns + [target_column]]

    if regression:
        kf = KFold(n_splits=10, shuffle=True, random_state=random_state+686)
        metric1 = mean_squared_error
        metricname1 = 'mse'
        metric2 = r2_score
        metricname2 = 'r2'
    else:
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state+686)
        metric1 = recall_score
        metricname1 = 'recall'
        metric2 = f1_score
        metricname2 = 'f1'
            
    scores = {'train': {metricname1: [], metricname2: []}, 'validation': {metricname1: [], metricname2: []}}

    for train_index, val_index in tqdm(kf.split(X = train_df.loc[:, predictor_columns], y = train_df.loc[:, target_column]), total=10):

        aux = train_df.reset_index(drop=True)
        X_train, X_val = aux.loc[train_index, predictor_columns], aux.loc[val_index, predictor_columns]
        y_train, y_val = aux.loc[train_index, target_column], aux.loc[val_index, target_column]


        """Preprocessing"""
        # Apply MinMax scaling only on training data (no need for unrotated RF)
        # scaler = MinMaxScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)

        # KNN imputing
        X_train_processed, X_val_processed = preprocess(X_train, X_val)

        # Train Random Forest
        if regression:
            model = RandomForestRegressor(random_state=random_state)
            model.fit(X_train_processed, y_train)
        else:
            model = RandomForestClassifier(random_state=random_state)
            model.fit(X_train_processed, y_train)
    
        # Predict on the single test instance
        y_train_pred = model.predict(X_train_processed) 
        y_val_pred = model.predict(X_val_processed)

        # sig = inspect.signature(metric2)
        if not regression:
            scores['train'][metricname1].append(metric1(y_train, y_train_pred, average='macro')) 
            scores['validation'][metricname1].append(metric1(y_val, y_val_pred, average='macro'))
            scores['train'][metricname2].append(metric2(y_train, y_train_pred, average='macro'))
            scores['validation'][metricname2].append(metric2(y_val, y_val_pred, average='macro'))
        else:
            scores['train'][metricname1].append(metric1(y_train, y_train_pred)) 
            scores['validation'][metricname1].append(metric1(y_val, y_val_pred))
            scores['train'][metricname2].append(metric2(y_train, y_train_pred))
            scores['validation'][metricname2].append(metric2(y_val, y_val_pred))


    # test_scaler = MinMaxScaler()
    # X_val_scaled = scaler.fit_transform(train_df.loc[:, predictor_columns])

    # test_scaler = MinMaxScaler()
    # X_test_scaled = scaler.transform(X_test)


    os.makedirs(f'/nfs/home/genovese/thesis-wildfire-genovese/outputs/random_forest/{results_file_name}/', exist_ok=True)

    with open(f'/nfs/home/genovese/thesis-wildfire-genovese/outputs/random_forest/{results_file_name}/{results_file_name}_scores.pkl', 'wb') as f:
        pickle.dump(scores, f)

    save_clean_data(X_test, 'X_test', f'/nfs/home/genovese/thesis-wildfire-genovese/outputs/random_forest/{results_file_name}/',
                    force=True)
    save_clean_data(pd.DataFrame(y_test, columns = [target_column]), 'y_test', f'/nfs/home/genovese/thesis-wildfire-genovese/outputs/random_forest/{results_file_name}/',
                    force=True)