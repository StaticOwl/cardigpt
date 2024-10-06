"""
File: ECGDataset.py
Project: potluck
Author: malli
Created: 06-10-2024
Description: write_a_description
"""

from .ECGData import ECGData
from .Transformers import Compose, RandomClip, Normalize, Retype, ValClip
import pandas as pd

nomalisetype = 'none'
start = 0
seq_length = 4096
sample_ratio = 0.5

data_transforms = {
    'train': Compose([
        RandomClip(len=seq_length),
        Normalize(nomalisetype),
        Retype()
    ]),
    'val': Compose([
        ValClip(len=seq_length),
        Normalize(nomalisetype),
        Retype()
    ]),
    'test': Compose([
        ValClip(len=seq_length),
        Normalize(nomalisetype),
        Retype()
    ])
}


class ECGDataset(object):
    num_classes = 24
    input_channel = 12

    def __init__(self, args):
        self.split = args.split if args.split else '0'
        self.data_dir = args.data_dir if args.data_dir else './data'

    def data_preprare(self, test=False):
        train_path = './data_split/train_split' + self.split + '.csv'
        val_path = './data_split/val_split' + self.split + '.csv'
        test_path = './data_split/test_split' + self.split + '.csv'

        train_pd = pd.read_csv(train_path)
        val_pd = pd.read_csv(val_path)

        train_dataset = ECGData(anno_pd=train_pd, transform=data_transforms['train'], data_dir=self.data_dir)
        val_dataset = ECGData(anno_pd=val_pd, transform=data_transforms['val'], data_dir=self.data_dir)
        test_dataset = None

        if test:
            test_pd = pd.read_csv(test_path)
            test_dataset = ECGData(anno_pd=test_pd, transform=data_transforms['test'], data_dir=self.data_dir)

        return train_dataset, val_dataset, test_dataset
