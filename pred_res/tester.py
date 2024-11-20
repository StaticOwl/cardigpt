"""
File: tester.py
Project: potluck
Author: staticowl
Created: 21-10-2024
Description: This module contains functions for processing ECG data, running classifiers, loading models, and testing models.
"""

import time
import os
import pandas as pd
import logging
from utils.predict_utils import load_model, load_test_data, run_classifier

logger = logging.getLogger(__name__)

def test(run_args):
    """
    Performs testing on the loaded model using ECG test data.

    Args:
        run_args: Arguments for running the test.
    """
    model_input = run_args.test_model_path
    test_dir = './input_data/test/'
    output_dir = './results/'

    input_files = []
    for f in os.listdir(test_dir):
        if os.path.isfile(os.path.join(test_dir, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    logger.info("Test model path: {}".format(model_input))
    model_all = load_model(model_input, run_args.model_name, run_args.test_model_name)
    num_files = len(input_files)

    df_pred = None

    for i, f in enumerate(input_files):
        logger.info("Predicting {} of {}".format(i + 1, num_files))
        tmp_input_file = os.path.join(test_dir, f)
        data, header_data = load_test_data(tmp_input_file)
        label, score, classes = run_classifier(data, header_data, model_all)
        if df_pred is None:
            columns = ['filename'] + ['{}_label'.format(c) for c in classes] + ['{}_score'.format(c) for c in classes]
            df_pred = pd.DataFrame(columns=columns)

        row_data = {
            'filename': f,
            **{'{}_label'.format(c): label[j] for j, c in enumerate(classes)},
            **{'{}_score'.format(c): score[j] for j, c in enumerate(classes)}
        }
        df_tmp = pd.DataFrame([row_data])
        df_pred = pd.concat([df_pred, df_tmp], ignore_index=True)

    df_pred.to_csv(os.path.join(output_dir, f'predictions-{time.time()}.csv'), index=False)

    logger.info("Predictions saved!")