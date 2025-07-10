import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def preprocess(X_train, X_val, model='', factor=1.5):

    # impute NA
    imputer = KNNImputer(n_neighbors=10, weights='distance')
    imputed_train = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    imputed_val = pd.DataFrame(imputer.transform(X_val), index=X_val.index, columns=X_val.columns)


    if model=='linear_regression':
        # remove outliers
        for column in imputed_train.columns:
            Q1 = imputed_train[column].quantile(0.25)
            Q3 = imputed_train[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            imputed_train = imputed_train[(imputed_train[column] >= lower_bound) & (imputed_train[column] <= upper_bound)]


    # scale variabels
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(imputed_train)
    X_val_scaled = scaler.transform(imputed_val)

    X_train_processed = pd.DataFrame(X_train_scaled, columns=imputed_train.columns, index=imputed_train.index)
    X_val_processed = pd.DataFrame(X_val_scaled, columns=imputed_val.columns, index=imputed_val.index)

    return X_train_processed, X_val_processed