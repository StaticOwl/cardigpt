"""
File: main.py
Author: Koustav Mallick
Date: 03/10/2024
Description: Main Driver Code
"""

import os, json, argparse
from logging_config import setup_logging
from utils import kaggle_data_downloader, load_and_process_data

import logging

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', default=False)
    args = parser.parse_args()
    with open(os.path.join(os.path.dirname(__file__), 'datadict.json')) as f:
        datadict = json.load(f)
    kaggle_data_downloader(datadict, args.download)

    gender, age, labels, ecg_filenames = load_and_process_data(datadict)

    for item in [gender, age, labels, ecg_filenames]:
        print(item)
        input("Press any keystroke...")
