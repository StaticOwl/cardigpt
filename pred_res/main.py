"""
File: main.py
Author: Koustav Mallick
Date: 03/10/2024
Description: Main Driver Code
"""

import argparse
import json
import logging
import os

from logging_config import setup_logging
from train import Trainer
from utils.datasplit import read_and_split_data
from utils.downloader import kaggle_data_downloader

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--prepare_train', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./trained_model',
                        help='the directory to save latest trained model')
    parser.add_argument('--split', type=str, default='0', help='The number of split')
    parser.add_argument('--data_dir', type=str, default='./data', help='the directory of the data')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--layers_num_last', type=int, default=0, help='the number of last layers which unfreeze')
    parser.add_argument('--lr', type=float, default=0.0003, help='the initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--step_size', type=int, default=3, help='stepLR step decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--max_epoch', type=int, default=5, help='max number of epoch')
    parser.add_argument('--win_length', type=int, default=4096, help='window length')
    parser.add_argument('--overlap', type=int, default=256, help='overlap')
    parser.add_argument('--log_step', type=int, default=10, help='step after log prints during training')
    parser.add_argument('--model_name', type=str, default='resnet', help='the name of the model')
    parser.add_argument('--patience', type=int, default=40, help='the patience for early stop')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='the min delta for early stop')
    parser.add_argument('--lambda_pd', type=float, default=1.0, help='Weight for Prototype Diversity Loss')
    parser.add_argument('--num_prototypes', type=int, default=10, help='Number of Prototypes')
    parser.add_argument('--prototype', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    setup_logging()
    args = parse_args()
    with open(os.path.join(os.path.dirname(__file__), 'datadict.json')) as f:
        datadict = json.load(f)
    kaggle_data_downloader(datadict, args.download)
    if args.download or args.prepare_train:
        read_and_split_data('./input_data/train/')

    for k, v in args.__dict__.items():
        logger.info("{}: {}".format(k, v))

    trainer = Trainer(args)
    trainer.setup()
    trainer.train()
