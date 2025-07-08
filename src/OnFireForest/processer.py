import sys 
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src")
from importlib import reload
import utils
reload(utils)
from utils import *

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def preprocess(X_train, X_val):

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    imputer = KNNImputer(n_neighbors=10, weights='distance')
    imputed_train = imputer.fit_transform(X_train_scaled)
    imputed_val = imputer.transform(X_val_scaled)

    X_train_processed = pd.DataFrame(imputed_train, columns=X_train.columns, index=X_train.index)
    X_val_processed = pd.DataFrame(imputed_val, columns=X_val.columns, index=X_val.index)

    return X_train_processed, X_val_processed