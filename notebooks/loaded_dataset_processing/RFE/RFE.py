import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import inspect
# import datetime
# import tensorflow as tf

import sys
sys.path.append(rf"/nfs/home/genovese/thesis-wildfire-genovese/src/OnFireForest/")
from importlib import reload
import processer
reload(processer)
from processer import preprocess



def adj_r2(y_true, y_pred, sample_size, n_variables):
    R2 = metrics.r2_score(y_true, y_pred)
    return 1-(1-R2)*(sample_size-1)/(sample_size-n_variables-1)

def classify(output: float) -> int:
    if output <= 0.2:
        return 0
    elif output <= 0.4:
        return 1
    elif output <= 0.6:
        return 2
    elif output <= 0.8:
        return 3
    else:
        return 4

def RecursiveFeatureSelection(X_train, X_val, y_train, y_val, model, c_y_train, c_y_val):

    high_score = 0                                      
    nof = 0

    current_features = X_train.columns
    rfe_tracking = pd.DataFrame(index=X_train.columns)
    results_rfe = pd.DataFrame(index=['train', 'validation'])

    iteration = 1
    # Loop to select the best no of features [RFE]
    while(len(current_features) > 3):
        print(f'Running iteration {iteration}, with {len(current_features)} features...')

        if hasattr(model, 'coef_'):
            X_train_processed, X_val_processed = preprocess(X_train.loc[:, current_features], X_val.loc[:, current_features], model='linear_regression')
        else:
            X_train_processed, X_val_processed = preprocess(X_train.loc[:, current_features], X_val.loc[:, current_features])
        rfe = RFE(estimator = model, step=1, verbose=5, n_features_to_select=0.9)
    
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_rfe_iteration_{iteration}"
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        y_train_processed = y_train.loc[X_train_processed.index]
        c_y_train_processed = c_y_train.loc[X_train_processed.index]
        rfe.fit(X_train_processed, y_train_processed)
        X_train_rfe = rfe.transform(X_train_processed)
        X_val_rfe = rfe.transform(X_val_processed)
        model.fit(X_train_rfe, y_train_processed)
    
        # Storing results on training data
        pred_train = model.predict(X_train_rfe)
        train_score = adj_r2(y_train_processed, pred_train, y_train_processed.shape[0], X_train_rfe.shape[1])
        c_train_score = metrics.accuracy_score(c_y_train_processed, [classify(y) for y in pred_train])
    
        # Storing results on validation data
        pred_val = model.predict(X_val_rfe)
        val_score = adj_r2(y_val, pred_val, y_val.shape[0], X_val_rfe.shape[1])
        c_val_score = metrics.accuracy_score(c_y_val, [classify(y) for y in pred_val])
    
        results_rfe[f'adjr2_iteration_{iteration}'] = [train_score, val_score]
        results_rfe[f'accuracy_iteration_{iteration}'] = [c_train_score, c_val_score]

        current_features = X_train_processed.columns[rfe.support_]
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError("Model doesn't provide feature importances.")

        rfe_tracking.loc[current_features, f'iteration_{iteration}'] = importances

        # Check best score
        if (np.abs(val_score) >= high_score):
            high_score = val_score
            nof = np.sum(rfe.support_)
       
        iteration += 1
    
    # Print the optimum number of features and the score with that number of features
    print("Optimum number of features: %d" %nof)
    print("Score with %d features: %f" % (nof, high_score))
    print("\nFeatures to select:")

    return results_rfe, rfe_tracking






