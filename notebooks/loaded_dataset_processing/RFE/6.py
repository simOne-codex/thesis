import RFE
from importlib import reload
reload(RFE)
from RFE import *
from math import sqrt

random_state = 92656

data_rekriged = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/not_imputed_dataset_kriged.csv').drop(columns='Unnamed: 0')
data_non_rekriged = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/not_imputed_dataset_non_kriged.csv').drop(columns='Unnamed: 0')

classification_logistic = LogisticRegression(n_jobs=-1, random_state=random_state, max_iter=300)
regression_linear = LinearRegression(n_jobs=-1)
regression_forest = RandomForestRegressor(n_jobs=-1, random_state=random_state, max_depth=int(sqrt(data_rekriged.shape[1])))
classification_forest = RandomForestClassifier(n_jobs=-1, random_state=random_state, max_depth=int(sqrt(data_rekriged.shape[1])))
for n, data in enumerate([data_rekriged]):
    
    string='rekriged'

    for m, model in tqdm(enumerate([regression_linear]), desc=f'Running {string} RFE...'):

        string2='regression'
        y='target'
        metric=adj_r2

        string3 = 'lr'

        
        explanatory = data.loc[:, [col for col in data.columns if col not in ['target', 'label', 'fire_id']]]
        target =  data.loc[:, y]
        X_train, X_test, y_train, y_test = train_test_split(explanatory,
                                                    target,
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=random_state)
        results_rfe, rfe_tracking = RecursiveFeatureSelection(X_train, X_test, y_train, y_test, model=model, metric=metric)
        results_rfe.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/outputs/rfe/{string}_{string2}_{string3}_rfe.csv')
        rfe_tracking.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/outputs/rfe/{string}_{string2}_{string3}_rfe_tracking.csv')

