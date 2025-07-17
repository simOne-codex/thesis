import RFE
from importlib import reload
reload(RFE)
from RFE import *
from math import sqrt

random_state = 92656

data_rekriged = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/not_imputed_dataset_rekriged.csv', index_col=0)
data_non_rekriged = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/not_imputed_dataset_non_rekriged.csv', index_col=0)
X_pca = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/pca_tranformed_without_target.csv')

data_non_rekriged = X_pca.assign(target=data_non_rekriged.target, label = data_non_rekriged.label)
data_rekriged = X_pca.assign(target=data_rekriged.target, label = data_rekriged.label)


classification_logistic = LogisticRegression(n_jobs=-1, random_state=random_state, max_iter=300)
regression_linear = LinearRegression(n_jobs=-1)
regression_forest = RandomForestRegressor(n_jobs=-1, random_state=random_state, max_depth=int(sqrt(data_rekriged.shape[1])))
classification_forest = RandomForestClassifier(n_jobs=-1, random_state=random_state, max_depth=int(sqrt(data_rekriged.shape[1])))
for n, data in enumerate([data_rekriged]):

    string='rekriged'

    for m, model in tqdm(enumerate([regression_forest]), desc=f'Running {string} RFE...'):

        string2='regression'
        y='target'
        metric=adj_r2
        string3 = 'rf'

        
        explanatory = data.loc[:, [col for col in data.columns if col not in ['target', 'label', 'fire_id', 'ALTIMETRIA_SUPERFICIE',
 'ALTIMETRIA_ALT_MIN',
 'ALTIMETRIA_ALT_MAX',
 'ALTIMETRIA_RANGE',
 'ALTIMETRIA_MEDIA',
 'ALTIMETRIA_MEDIANA',
 'ALTIMETRIA_STD',
 'ALTIMETRIA_zona_altimetrica']]]
        target =  data.loc[:, y]
        X_train, X_test, y_train, y_test = train_test_split(explanatory,
                                                    target,
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=random_state)
        results_rfe, rfe_tracking = RecursiveFeatureSelection(X_train, X_test, y_train, y_test, model=model,
                                                              c_y_train = data.loc[X_train.index, 'label'], c_y_val = data.loc[X_test.index, 'label'])
        results_rfe.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/outputs/rfe_pca/{string}_{string2}_{string3}_rfe.csv')
        rfe_tracking.to_csv(f'/nfs/home/genovese/thesis-wildfire-genovese/outputs/rfe_pca/{string}_{string2}_{string3}_rfe_tracking.csv')

