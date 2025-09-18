import RFE
from importlib import reload
reload(RFE)
from RFE import *
from math import sqrt
from sklearn.metrics import accuracy_score

random_state = 92656

X_train_val = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/only_human_study/database/sample_b/X_train_val.csv')
y_train_val = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/only_human_study/database/sample_b/y_train_val.csv')

regression_forest = RandomForestRegressor(n_jobs=-1, random_state=random_state, max_depth=int(sqrt(X_train_val.shape[1])))
classification_forest = RandomForestClassifier(n_jobs=-1, random_state=random_state, max_depth=int(sqrt(X_train_val.shape[1])))

for m, model in tqdm(enumerate([regression_forest, classification_forest])):

    if m == 0:
        string2='regression'
        y='target'
        metric=adj_r2
    else:
        string2 = 'classification'
        y='target'
        metric=accuracy_score

    
    explanatory = X_train_val
    target =  y_train_val
    X_train, X_val, y_train, y_val = train_test_split(explanatory,
                                                target,
                                                test_size=0.15,
                                                shuffle=True,
                                                random_state=random_state,
                                                stratify=target)
    results_rfe, rfe_tracking = RecursiveFeatureSelection(X_train, X_val, y_train.iloc[:, 0], y_val.iloc[:, 0], model=model,
                                                            c_y_train = False, c_y_val = False)
    results_rfe.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/only_human_study/outputs/sample_b/{string2}_rf_rfe.csv')
    rfe_tracking.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/only_human_study/outputs/sample_b/{string2}_rf_rfe_tracking.csv')

