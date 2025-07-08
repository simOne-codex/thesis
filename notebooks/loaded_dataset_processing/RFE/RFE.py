import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import inspect
import datetime
import tensorflow as tf

import sys
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src/OnFireForest/")
from importlib import reload
import processer
reload(processer)
from processer import preprocess



def adj_r2(y_true, y_pred, sample_size, n_variables):
    R2 = metrics.r2_score(y_true, y_pred)
    return 1-(1-R2)*(sample_size-1)/(sample_size-n_variables-1)


def RecursiveFeatureSelection(X_train, X_val, y_train, y_val, model, metric):

    high_score = 0                                      
    nof = 0

    current_features = X_train.columns
    rfe_tracking = pd.DataFrame(index=X_train.columns)
    results_rfe = pd.DataFrame(index=['train', 'validation'])

    iteration = 1
    # Loop to select the best no of features [RFE]
    while(len(current_features) > 3):
        print(f'Running iteration {iteration}, with {len(current_features)} features...')

        X_train_processed, X_val_processed = preprocess(X_train.loc[:, current_features], X_val.loc[:, current_features])

        rfe = RFE(estimator = model, step=1, verbose=5)
    
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_rfe_iteration_{iteration}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        rfe.fit(X_train_processed, y_train, callbacks=[tensorboard_callback])
        X_train_rfe = rfe.transform(X_train_processed)
        X_val_rfe = rfe.transform(X_val_processed)
        model.fit(X_train_rfe, y_train)
    
        # Storing results on training data
        pred_train = model.predict(X_train_rfe)
        sig = inspect.signature(metric)
        if 'average' in sig.parameters:
            train_score = metric(y_train, pred_train, average='macro')
        else:
            train_score = metric(y_train, pred_train, y_train.shape[0], X_train_processed.shape[1])
    
        # Storing results on validation data
        pred_val = model.predict(X_val_rfe)
        if 'average' in sig.parameters:
            val_score = metric(y_val, pred_val, average='macro')
        else:
            val_score = metric(y_val, pred_val, y_val.shape[0], X_train_processed.shape[1])
    
        results_rfe[f'iteration_{iteration}'] = [train_score, val_score]

        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError("Model doesn't provide feature importances.")

        # Check best score
        if (val_score >= high_score):
            high_score = val_score
            nof = np.sum(rfe.support_)
        
        # rfe_tracking[f'iteration_{iteration}'] = [0]*rfe_tracking.shape[0]

        current_features = X_train_processed.columns[rfe.support_]
        for c in range(4):
            rfe_tracking.loc[current_features, f'iteration_{iteration}_class_{c}'] = importances[c]

        iteration += 1
    
    # Print the optimum number of features and the score with that number of features
    print("Optimum number of features: %d" %nof)
    print("Score with %d features: %f" % (nof, high_score))
    print("\nFeatures to select:")

    return results_rfe, rfe_tracking






