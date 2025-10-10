import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score


#### TRAIN TEST SPLIT
random_state = 8856
transform = {0.0: 0,
 0.1: 1,
 0.2: 2,
 0.3: 3,
 0.4: 4,
 0.5: 5,
 0.6: 6,
 0.7: 7,
 0.8: 8,
 0.9: 9,
 1.0: 10}

r = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/cleaned_final_rekriged.csv').rename(columns={'Unnamed: 0': 'fire_id'})
nonr = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/cleaned_final_non_rekriged.csv').rename(columns={'Unnamed: 0': 'fire_id'})

r['target_encoded'] = r.target.round(1).map(transform)
nonr['target_encoded'] = nonr.target.round(1).map(transform)

explanatory_rekriged = r.loc[:, [col for col in r.columns if col not in ['target', 'label', 'target_encoded']]]
target_rekriged =  r.loc[:, ['target_encoded']]
X_train, X_test, y_train, y_test = train_test_split(explanatory_rekriged,
                                                    target_rekriged,
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=random_state,
                                                    stratify=target_rekriged
                                                    )

pd.DataFrame(X_train, columns = explanatory_rekriged.columns).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/X_train_val_rekriged.csv')
pd.DataFrame(y_train, columns = target_rekriged.columns).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/y_train_val_rekriged.csv')

pd.DataFrame(X_test, columns = explanatory_rekriged.columns).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/X_test_rekriged.csv')
pd.DataFrame(y_test, columns = target_rekriged.columns).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/y_test_rekriged.csv')


explanatory_non_rekriged = nonr.loc[:, [col for col in nonr.columns if col not in ['target', 'label', 'target_encoded']]]
target_non_rekriged =  nonr.loc[:, ['target_encoded']]
X_train, X_test, y_train, y_test = train_test_split(explanatory_non_rekriged,
                                                    target_non_rekriged,
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=random_state,
                                                    stratify=target_non_rekriged
                                                    )

pd.DataFrame(X_train, columns = explanatory_non_rekriged.columns).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/X_train_val_non_rekriged.csv')
pd.DataFrame(y_train, columns = target_non_rekriged.columns).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/y_train_val_non_rekriged.csv')

pd.DataFrame(X_test, columns = explanatory_non_rekriged.columns).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/X_test_non_rekriged.csv')
pd.DataFrame(y_test, columns = target_non_rekriged.columns).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/y_test_non_rekriged.csv')
####



random_state = 92656
cv_rs=1633


#### REKRIGED GRIDSEARCH
X_train_r = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/X_train_val_rekriged.csv').set_index('fire_id', drop=True)
y_train_r = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/y_train_val_rekriged.csv').set_index(X_train_r.index).iloc[:, 1]
X_train_r.drop(columns=X_train_r.columns[0], inplace=True)

regression_forest = RandomForestRegressor(n_jobs=-1, random_state=random_state)
param_grid={'max_depth':[int(np.sqrt(X_train_r.shape[1])), 50, 100], 'min_samples_leaf': [1, 5, 10]}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=cv_rs)
# cv=10

gs_r = GridSearchCV(regression_forest,
                  param_grid=param_grid,
                  cv=cv, 
                  verbose=5, 
                  n_jobs=-1, 
                  scoring='r2')

gs_r.fit(X_train_r, y_train_r)

pd.DataFrame(gs_r.cv_results_).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/outputs/grid_searches/model_fine_tuning_rekriged.csv')
####


#### NON REKRIGED GRIDSEARCH
X_train_nonr = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/X_train_val_non_rekriged.csv').set_index('fire_id')
y_train_nonr = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/y_train_val_non_rekriged.csv').set_index(X_train_nonr.index).iloc[:, 1]
X_train_nonr.drop(columns=X_train_nonr.columns[0], inplace=True)

regression_forest = RandomForestRegressor(n_jobs=-1, random_state=random_state)
param_grid={'max_depth':[int(np.sqrt(X_train_nonr.shape[1])), 50, 100], 'min_samples_leaf': [1, 5, 10]}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=cv_rs)
# cv=10

gs_nonr = GridSearchCV(regression_forest,
                  param_grid=param_grid,
                  cv=cv, 
                  verbose=5, 
                  n_jobs=-1, 
                  scoring='r2')

gs_nonr.fit(X_train_nonr, y_train_nonr)

pd.DataFrame(gs_nonr.cv_results_).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/outputs/grid_searches/model_fine_tuning_non_rekriged.csv')
####


########################## CLASSIFIER 

#### REKRIGED GRIDSEARCH
X_train_r = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/X_train_val_rekriged.csv').set_index('fire_id', drop=True)
y_train_r = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/y_train_val_rekriged.csv').set_index(X_train_r.index).iloc[:, 1]
X_train_r.drop(columns=X_train_r.columns[0], inplace=True)

regression_forest = RandomForestClassifier(n_jobs=-1, random_state=random_state)
param_grid={'max_depth':[int(np.sqrt(X_train_r.shape[1])), 50, 100], 'min_samples_leaf': [1, 5, 10]}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=cv_rs)
# cv=10

gs_r = GridSearchCV(regression_forest,
                  param_grid=param_grid,
                  cv=cv, 
                  verbose=5, 
                  n_jobs=-1, 
                  scoring='accuracy')

gs_r.fit(X_train_r, y_train_r)

pd.DataFrame(gs_r.cv_results_).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/outputs/grid_searches/model_classification_fine_tuning_rekriged.csv')
####


#### NON REKRIGED GRIDSEARCH
X_train_nonr = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/X_train_val_non_rekriged.csv').set_index('fire_id')
y_train_nonr = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/model_input/y_train_val_non_rekriged.csv').set_index(X_train_nonr.index).iloc[:, 1]
X_train_nonr.drop(columns=X_train_nonr.columns[0], inplace=True)

regression_forest = RandomForestClassifier(n_jobs=-1, random_state=random_state)
param_grid={'max_depth':[int(np.sqrt(X_train_nonr.shape[1])), 50, 100], 'min_samples_leaf': [1, 5, 10]}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=cv_rs)
# cv=10

gs_nonr = GridSearchCV(regression_forest,
                  param_grid=param_grid,
                  cv=cv, 
                  verbose=5, 
                  n_jobs=-1, 
                  scoring='accuracy')

gs_nonr.fit(X_train_nonr, y_train_nonr)

pd.DataFrame(gs_nonr.cv_results_).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/ablation_study_zero_image/outputs/grid_searches/model_classification_fine_tuning_non_rekriged.csv')
####