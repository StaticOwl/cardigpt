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
    parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
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
    
    return parser.parse_args()


def train_main(run_args):
    """
    Main function for training the model.
    
    Args:
        run_args (argparse.Namespace): Parsed arguments for training
    """
    setup_logging()
    
    with open(os.path.join(os.path.dirname(__file__), 'datadict.json')) as f:
        datadict = json.load(f)
        
    kaggle_data_downloader(datadict, run_args.download)
    
    if run_args.download or run_args.prepare_train:
        read_and_split_data('./input_data/train/')
        
    for k, v in run_args.__dict__.items():
        logger.info("{}: {}".format(k, v))
        
    trainer = Trainer(run_args)
    trainer.setup()
    trainer.train()


def test_main(run_args):
    """
    Main function for testing the model.
    
    Args:
        run_args (argparse.Namespace): Parsed arguments for testing
    """
    setup_logging()
    test(run_args)


if __name__ == '__main__':
    args = parse_args()

    if args.type == 'train':
        train_main(args)
    elif args.type == 'test':
        test_main(args)
    elif args.type == 'predict':
        raise NotImplementedError('predict is not implemented yet')
