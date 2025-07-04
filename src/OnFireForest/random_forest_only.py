import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, recall_score, f1_score
from tqdm import tqdm

import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *


def run_rf_only(input_csv_path: str, output_model_pkl_path: str, regression: bool, results_file_name: str, random_state: int = 3469) -> None:

    df = pd.read_csv(input_csv_path).drop_duplicates()

    if regression:
        y = 'target'
    else:
        y='label'


    non_train_columns = ['target', 'geometry', 'YYYY', 'DD', 'MM', 'day', 'id', 'fire_id', 'label']
    target_column = y
    predictor_columns = [col for col in df.columns if col not in non_train_columns]

    test_df = df.sample(frac=0.15, random_state = random_state + 185)
    test_indices = test_df.index
    X_test = test_df.loc[:, predictor_columns]
    y_test = test_df.loc[:, target_column]

    train_indices = [idx for idx in df.index if idx not in test_indices]
    train_df = df.loc[train_indices, predictor_columns + [target_column]]



# leave-one-out cross validation since the dataset is small
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    with open(output_model_pkl_path, 'wb') as f:
        for train_index, test_index in tqdm(loo.split(train_df)):

            X_train, X_test = train_df.loc[train_index, predictor_columns], train_df.loc[test_index, predictor_columns]
            y_train, y_test = train_df[train_index, target_column], train_df.loc[test_index, target_column]

            # Apply MinMax scaling only on training data (no need for unrotated RF)
            # scaler = MinMaxScaler()
            # X_train_scaled = scaler.fit_transform(X_train)
            # X_test_scaled = scaler.transform(X_test)

            # Train Random Forest Regressor
            if regression:
                model = RandomForestRegressor(random_state=random_state)
                model.fit(X_train, y_train)
            else:
                model = RandomForestClassifier(random_state=random_state)

            # Predict on the single test instance
            y_pred.append(model.predict(X_test)[0])
            y_true.append(y_test[0])

        pickle.dump(model, f)


    # test_scaler = MinMaxScaler()
    # X_val_scaled = scaler.fit_transform(train_df.loc[:, predictor_columns])

    if regression:
        metric1 = mean_squared_error
        metricname1 = 'mse'
        metric2 = r2_score
        metricname2 = 'r2'
    else:
        metric1 = recall_score
        metricname1 = 'recall'
        metric2 = f1_score
        metricname2 = 'f1'
    
    train_score_1 = metric1(y_true, y_pred)
    train_score_2 = metric2(y_true, y_pred)

    # test_scaler = MinMaxScaler()
    # X_test_scaled = scaler.transform(X_test)

    test_score_1 = metric1(y_test, model.predict(X_test))
    test_score_2 = metric2(y_test, model.predict(X_test))

    results = pd.DataFrame({metricname1: {'train': train_score_1, 'test': test_score_1}, metricname2: {'train': train_score_2, 'test': test_score_2}})
    save_clean_data(results, results_file_name, '/nfs/home/genovese/thesis-wildfire-genovese/outputs/random_forest/')