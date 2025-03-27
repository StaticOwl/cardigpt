"""
File: main.py
Author: Koustav Mallick
Date: 03/10/2024
Description: Main Driver Code for training and testing models
"""

import argparse
import json
import logging
import os

from logging_config import setup_logging
from tester import test
from predict import predict
from train import Trainer
from utils.datasplit import read_and_split_data
from utils.downloader import kaggle_data_downloader

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments for training, testing, and prediction tasks.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['train', 'test', 'predict'], help='train or test')
    parser.add_argument('--test_model_path', type=str, default='./model_repo/', help='the path of the model to be tested')
    parser.add_argument('--test_model_name', type=str, help='the name of the model to be tested')
    parser.add_argument('--download', action='store_true', help='download data')
    parser.add_argument('--early_stop', action='store_true', help='enable early stopping')
    parser.add_argument('--prepare_train', action='store_true', help='prepare training data')
    parser.add_argument('--save_dir', type=str, default='./trained_model', help='directory to save latest trained model')
    parser.add_argument('--split', type=str, default='0', help='The number of split')
    parser.add_argument('--data_dir', type=str, default='./data', help='directory of the data')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='number of training process workers')
    parser.add_argument('--layers_num_last', type=int, default=0, help='number of last layers to unfreeze')
    parser.add_argument('--lr', type=float, default=0.00003, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--step_size', type=int, default=3, help='stepLR step decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--max_epoch', type=int, default=5, help='max number of epochs')
    parser.add_argument('--win_length', type=int, default=4096, help='window length')
    parser.add_argument('--overlap', type=int, default=256, help='overlap')
    parser.add_argument('--log_step', type=int, default=10, help='step after log prints during training')
    parser.add_argument('--model_name', type=str, default='resnet', help='the name of the model')
    parser.add_argument('--patience', type=int, default=40, help='patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='minimum delta for early stopping')
    parser.add_argument('--lambda_pd', type=float, default=1.0, help='Weight for Prototype Diversity Loss')
    parser.add_argument('--num_prototypes', type=int, default=10, help='Number of Prototypes')
    parser.add_argument('--prototype', action='store_true', help='enable prototype')
    parser.add_argument('--pretrained', action='store_true', help='enable pretrained model')
    parser.add_argument('--csv', action='store_true', help='save predictions as csv')
    parser.add_argument('--input_dir', type=str, default='./input_data/test/', help='directory of the input data')
    parser.add_argument('--test_file', type=str, help='name of the input file')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    setup_logging()
    if args.type == 'train':
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

    elif args.type == 'test':
        test(args)

    elif args.type == 'predict':
        predict(args)
