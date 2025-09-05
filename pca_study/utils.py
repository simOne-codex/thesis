import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer


from sklearn.model_selection import train_test_split




def train_test_split_before_pca(df, output_folder, random_state, impute=True):

    explanatory = df.loc[:, [col for col in df.columns if col not in ['target']]]
    target =  df.loc[:, 'target']
    X_train, X_test, y_train, y_test = train_test_split(explanatory,
                                                    target,
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=random_state,
                                                    stratify=target)


    if impute:
        imputer = KNNImputer(n_neighbors=10, weights='distance')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    columns_to_drop = [col for col in X_train.columns if X_train[col].nunique() == 1]
    X_train.drop(columns = columns_to_drop, inplace=True)
    X_test.drop(columns = columns_to_drop, inplace=True)

    pd.DataFrame(X_train).to_csv(output_folder + 'X_train_val.csv', index=False)
    pd.DataFrame(y_train).to_csv(output_folder + 'y_train_val.csv', index=False)
    
    pd.DataFrame(X_test).to_csv(output_folder + 'X_test.csv', index=False)
    pd.DataFrame(y_test).to_csv(output_folder + 'y_test.csv', index=False)



def pca_with_results(df, plot=True):
       
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Step 2: Apply PCA
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)

    # Step 3: Create a DataFrame of PCA results
    pca_df = pd.DataFrame(data=pca_data, columns=[f'PC{c}' for c in range(1, df.shape[1]+1)], index=df.index)

    # Step 4: Explained variance
    explained_variance = pca.explained_variance_ratio_
    pcs = np.arange(1, len(explained_variance) + 1)

    # Step 5: Plot
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(10,8))
        ax[0].plot(pcs[:81], explained_variance[:81], marker='o', linestyle='-')
        ax[0].set_title('Explained Variance by Principal Component')
        ax[0].set_xlabel('Principal Component')
        ax[0].set_ylabel('Explained Variance Ratio')
        ax[0].set_xticks(range(1, 81, 4), range(1, 81, 4), rotation='vertical', fontsize=8)
        ax[0].grid(True)

        ax[1].plot(pcs[:81], np.cumsum(explained_variance[:81]), marker='o', linestyle='-')
        ax[1].set_title('Cumulative Explained Variance by Principal Component')
        ax[1].set_xlabel('Principal Component')
        ax[1].set_ylabel('Explained Variance Ratio')
        ax[1].set_xticks(range(1, 81, 4), range(1, 81, 4), rotation='vertical', fontsize=8)
        ax[0].grid(True)
        
        plt.subplots_adjust(wspace=0.5)
        plt.show()

    return scaler, pca, pca_df


def transform_test_after_pca_selection(X_test, scaler, pca, pc_to_select, output_folder):


    pca_test = pca.transform(scaler.transform(X_test))
    pca_test_df = pd.DataFrame(data=pca_test, columns=[f'PC{c}' for c in range(1, X_test.shape[1]+1)], index=X_test.index)
    selected_test = pca_test_df.loc[:, [f'PC{n}' for n in range(1, pc_to_select+1, 1)]]

    selected_test.to_csv(output_folder + 'final_X_test.csv', index=False)