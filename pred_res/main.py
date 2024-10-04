"""
File: main.py
Author: Koustav Mallick
Date: 03/10/2024

Description: Main Driver Code
"""

import numpy as np, os, shutil, json
from scipy.io import loadmat
from kaggle.api.kaggle_api_extended import KaggleApi

def kaggle_data_downloader(flag = False):
    api = KaggleApi()
    api.authenticate()

    if flag:
        with open(os.path.join(os.path.dirname(__file__), 'datadict.json')) as f:
            datadict = json.load(f)

        if os.path.exists('input_data'):
            shutil.rmtree('input_data')
        os.makedirs('input_data')
        
        for key, value in datadict.items():
            print(key, value)
            if key not in os.listdir('input_data'):
                api.dataset_download_files(value['source'], path=f'input_data/{key}', unzip=True)
                folder_path = os.path.join('input_data', key, value['inside_folder'])
                for filename in os.listdir(folder_path):
                    os.rename(os.path.join(folder_path, filename), os.path.join('input_data', key, filename))
                shutil.rmtree(folder_path)
            datadict[key]['path'] = os.path.join('input_data', key)

        with open('datadict.json', 'w') as f:
            json.dump(datadict, f)

if __name__ == '__main__':
    kaggle_data_downloader(True)

# def load_data(filename):
#     """
#     This function loads the data from the .mat files provided in the MIT-BIH dataset.
#     It takes the filename as an argument and returns the data and header data associated with the file.

#     Parameters:
#     filename (str): The name of the .mat file to be loaded.

#     Returns:
#     data (numpy.ndarray): The data loaded from the file.
#     header_data (list): The header data associated with the file.
#     """

#     # Load the data from the .mat file
#     x = loadmat(filename)
#     data = np.asarray(x['val'], dtype=np.float64)

#     # Create the new filename by replacing the .mat extension with .hea
#     new_file = filename.replace('.mat', '.hea')
#     input_header_file = os.path.join(new_file)

#     # Read the header data from the .hea file
#     with open(input_header_file, 'r') as f:
#         header_data = f.readlines()

#     # Return the data and header data
#     return data, header_data
