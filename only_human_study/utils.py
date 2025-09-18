import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


def train_test_split_before_rfe(df, output_folder, random_state, impute=True):

    explanatory = df.loc[:, [col for col in df.columns if col not in ['target']]]
    target =  df.loc[:, 'target']
    X_train, X_test, y_train, y_test = train_test_split(explanatory,
                                                    target,
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=random_state,
                                                    stratify=target)


    if impute:
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(X_train)
        scaled_test = scaler.transform(X_test)
        imputer = KNNImputer(n_neighbors=10, weights='distance')
        X_train = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(scaled_train)), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(scaler.inverse_transform(imputer.transform(scaled_test)), index=X_test.index, columns=X_test.columns)
    
    columns_to_drop = [col for col in X_train.columns if X_train[col].nunique() == 1]
    X_train.drop(columns = columns_to_drop, inplace=True)
    X_test.drop(columns = columns_to_drop, inplace=True)

    pd.DataFrame(X_train).to_csv(output_folder + 'X_train_val.csv', index=False)
    pd.DataFrame(y_train).to_csv(output_folder + 'y_train_val.csv', index=False)
    
    pd.DataFrame(X_test).to_csv(output_folder + 'X_test.csv', index=False)
    pd.DataFrame(y_test).to_csv(output_folder + 'y_test.csv', index=False)