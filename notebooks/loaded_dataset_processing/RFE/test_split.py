import pandas as pd
from sklearn.model_selection import train_test_split

random_state = 684

data_rekriged = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/not_imputed_dataset_kriged.csv').drop(columns='Unnamed: 0')
data_non_rekriged = pd.read_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/not_imputed_dataset_non_kriged.csv').drop(columns='Unnamed: 0')


explanatory_rekriged = data_rekriged.loc[:, [col for col in data_rekriged.columns if col not in ['taget', 'label']]]
target_rekriged =  data_rekriged.loc[:, ['target', 'label']]
X_train, X_test, y_train, y_test = train_test_split(explanatory_rekriged,
                                                    target_rekriged,
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=random_state)
pd.DataFrame(X_test).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/X_test_rekriged.csv')
pd.DataFrame(y_test).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/y_test_rekriged.csv')


explanatory_non_rekriged = data_non_rekriged.loc[:, [col for col in data_non_rekriged.columns if col not in ['taget', 'label']]]
target_non_rekriged =  data_non_rekriged.loc[:, ['target', 'label']]
X_train, X_test, y_train, y_test = train_test_split(explanatory_non_rekriged,
                                                    target_non_rekriged,
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=random_state)
pd.DataFrame(X_test).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/X_test_non_rekriged.csv')
pd.DataFrame(y_test).to_csv('/nfs/home/genovese/thesis-wildfire-genovese/database/model_input/y_test_non_rekriged.csv')