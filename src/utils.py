import os
# directory = rf"C:\Users\simof\Documents\GitHub\thesis"
directory = rf"/nfs/home/genovese/thesis-wildfire-genovese"
os.chdir(directory)

import shutil
import geopandas as gpd
import glob
import zipfile
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gsp
from unidecode import unidecode
from tqdm import tqdm

def extract_zipfile(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print("Inner file extracted ", os.listdir(extract_to_path))



extensions = {'DataFrame': '.csv', 'GeoDataFrame': '.geojson'}

def format_output_file(file, output_path, overwrite=False):
    filetype = file.__class__.__name__
    complete_filename = f"{output_path}{extensions[filetype]}"
    if overwrite:
        if extensions[filetype] == '':
            shutil.rmtree(complete_filename)
        else:
            os.remove(complete_filename)
    if filetype == 'DataFrame':
        file.to_csv(complete_filename, index=False)
    elif filetype == 'GeoDataFrame':
        file.to_file(complete_filename, index=False, driver ='GeoJSON')
    else:
        raise('Format not supported, file type: ', filetype)



def save_clean_data(file, filename, output_folder = rf'/nfs/home/genovese/thesis-wildfire-genovese/data/clean_data', force=False):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = f"{output_folder}/{filename}"

    if not os.path.exists(f"{output_path}{extensions[file.__class__.__name__]}"):
        format_output_file(file, output_path)
        print(f"File saved to: {output_path}")
    else: 
        print(f"File {filename} already saved")
        if force:
            user_response = 'y'
        else:
            user_response = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if user_response == 'y':
        # Proceed with overwriting the file
            format_output_file(file, output_path, overwrite=True)
            print("File has been overwritten")
        else:
            print("File not overwritten")
            



def get_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
    return outliers


def histbox(var, color='blue', xlabel=None, title=None):
    fig = plt.figure(figsize=(6,6))
    fig = plt.figure(figsize=(8, 6))
    gs = gsp.GridSpec(2, 1, height_ratios=[1, 3])

    ax_box = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    sns.histplot(var, kde=True, ax=ax_hist, color=color)
    sns.boxplot(x=var, ax=ax_box, color=color)

    ax_box.set(xlabel='', ylabel='')
    ax_box.set_xticks([], [])
    ax_box.set_yticks([], [])

    m = np.mean(var)
    ax_hist.axvline(m, linestyle='--', linewidth=1, label=rf'$\mathbf{{Mean}}$ = {m}')
    min_outl = np.min(get_outliers(var))
    ax_hist.axvline(min_outl, linestyle='--', linewidth=1, color='lightblue', label=rf'$\mathbf{{First \, outlier}}$ = {min_outl}')
    med = np.median(var)
    ax_hist.axvline(med, linestyle='--', linewidth=1, color='black', label=rf'$\mathbf{{Median}}$ = {med}')


    ax_hist.set_xlabel(xlabel, fontsize=8, fontweight='bold') 
    ax_hist.set_ylabel('Frequency', fontsize=8, fontweight='bold')
    ax_hist.legend(loc = 'upper right', fontsize=6, title='statistics', 
               title_fontproperties={'weight':'bold', 'size':'6'},
               frameon=False) # to remove the framebox
    
    sns.despine(top=True, right=True, ax=ax_hist)
    sns.despine(top=True, right=True, bottom=True, ax=ax_box)

    plt.suptitle(title, fontweight='bold', fontsize=10, y=.95)
    plt.show()


lettere_strane = {
    "a": ['à', 'á', 'â', 'â', 'ã', 'å', 'ā'],
    "e": ['è', 'é', 'ê', 'ë', 'ē', 'ĕ'], 
    "i": ['ì', 'í', 'î', 'ï', 'ī', 'ĩ'],
    "o": ['ò', 'ó', 'ô', 'ö', 'õ', 'ō'],
    "u": ['ù', 'ú', 'û', 'ü', 'ũ', 'ū'],
    "A": ['À', 'Á', 'Â', 'Ä', 'Ã', 'Å'],
    "E": ['È', 'É', 'Ê', 'Ë'],
    "I": ['Ì', 'Í', 'Î', 'Ï'],
    "O": ['Ò', 'Ó', 'Ô', 'Ö', 'Õ'], 
    "U": ['Ù', 'Ú', 'Û', 'Ü', 'Ũ']
}


def uniform_strings(row, var):
    # for normale, strane in lettere_strane.items():
    #     for strana in strane:
    #         row[var]= row[var].replace(strana, normale)
    row[var] = unidecode(row[var])
    row[var] = row[var].replace(' ', '').lower()
    return row

def check_strange_letters(df, var):
    return(df[~df[var].str.contains(r'^[A-Za-z\s\-\_\']+$')])


def separate_date(df, col):
    df[col] = pd.to_datetime(df[col])
    df['YYYY'] = df[col].dt.year
    df['MM'] = df[col].dt.month
    df['DD'] = df[col].dt.day
    return df

def set_city_code(row, var):
    if not str(row[var]).startswith('A'):
        row[var] = 'A' + str(row[var]).zfill(6)
    else:
        print('Format already uniformed')
    return row