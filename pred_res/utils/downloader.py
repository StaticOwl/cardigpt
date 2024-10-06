"""
File: downloader.py
Project: potluck
Author: malli
Created: 06-10-2024
Description: write_a_description
"""
import json
import logging
import os
import shutil

from kaggle import KaggleApi
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


def kaggle_data_downloader(datadict, flag=False):
    api = KaggleApi()
    api.authenticate()

    if flag:
        logger.info('Starting data download process')

        # Clear existing data directory
        if os.path.exists('input_data'):
            shutil.rmtree('input_data')
        os.makedirs('input_data')

        # Create train and test directories globally
        train_dir = os.path.join('input_data', 'train')
        test_dir = os.path.join('input_data', 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Download and process each dataset
        for key, value in tqdm(datadict.items(), desc='Downloading data'):
            dataset_path = 'input_data'

            if key not in os.listdir(dataset_path):
                api.dataset_download_files(value['source'], path=dataset_path, unzip=True)
                folder_path = os.path.join(dataset_path, value['inside_folder'])

                # Move files from inside folder to the main dataset path
                for filename in os.listdir(folder_path):
                    os.rename(os.path.join(folder_path, filename), os.path.join(dataset_path, filename))
                shutil.rmtree(folder_path)

            # Split data into train and test sets
            all_files = [f for f in os.listdir(dataset_path) if f not in ['train', 'test']]
            train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

            # Move files to respective directories
            for file in train_files:
                shutil.move(os.path.join(dataset_path, file), os.path.join(train_dir, file))
            for file in test_files:
                shutil.move(os.path.join(dataset_path, file), os.path.join(test_dir, file))

        # Save updated datadict with paths (optional)
        with open('datadict.json', 'w') as f:
            json.dump(datadict, f)

        logger.info('Finished downloading and splitting data')