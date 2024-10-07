"""
File: downloader.py
Project: potluck
Author: malli
Created: 06-10-2024
Description: Download data from Kaggle and split it into train and test sets
"""

import json
import logging
import os
import shutil
from pathlib import Path

from kaggle import KaggleApi
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


def kaggle_data_downloader(datadict, flag=False):
    """
    Download data from Kaggle and split it into train and test sets.

    Parameters
    ----------
    datadict : dict
        Dictionary containing dataset information
    flag : bool
        Flag to indicate whether to download and split the data
    """
    api = KaggleApi()
    api.authenticate()

    if flag:
        logger.info('Starting data download process')

        # Clear existing data directory
        data_dir = Path('input_data')
        if data_dir.exists():
            logger.info('Removing existing data directory')
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create train and test directories globally
        train_dir = data_dir / 'train'
        test_dir = data_dir / 'test'
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        # Download and process each dataset
        for key, value in tqdm(datadict.items(), desc='Downloading data'):
            dataset_path = data_dir

            if key not in dataset_path.iterdir():
                logger.info('Downloading dataset {}'.format(key))
                api.dataset_download_files(value['source'], path=dataset_path, unzip=True, quiet=False)
                folder_path = dataset_path / value['inside_folder']

                # Move files from inside folder to the main dataset path
                logger.info('Moving files from {} to {}'.format(folder_path, dataset_path))
                for file in folder_path.iterdir():
                    file.rename(dataset_path / file.name)
                shutil.rmtree(folder_path)

            # Split data into train and test sets
            all_files = [f for f in dataset_path.iterdir() if f.name not in ['train', 'test']]
            grouped_files = {}
            for file in all_files:
                basename, _ = os.path.splitext(file.name)
                if basename not in grouped_files:
                    grouped_files[basename] = []
                grouped_files[basename].append(file)

            train_group, test_group = train_test_split(list(grouped_files.values()), test_size=0.1, random_state=42)

            # Move files to respective directories
            for group in train_group:
                for file in group:
                    file.rename(train_dir / file.name)
            for group in test_group:
                for file in group:
                    file.rename(test_dir / file.name)

        # Save updated datadict with paths (optional)
        if flag:
            with open('datadict.json', 'w') as f:
                json.dump(datadict, f)

            logger.info('Finished downloading and splitting data')