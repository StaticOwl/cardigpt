"""
File: predict.py
Author: Koustav Mallick
Date: 06/11/2024

Description: [Add a brief description of the file here]
"""
import os
import logging
import json
from collections import defaultdict
from utils.predict_utils import load_model, load_test_data, run_classifier, json_serializer

logger = logging.getLogger(__name__)

def predict(run_args):
    """
    Performs prediction on the loaded model using ECG test data.

    Args:
        run_args: Arguments for running the test.
    """
    model_input = run_args.test_model_path
    test_dir = './input_data/test/'
    filename = run_args.test_file

    logger.info("Test model path: {}".format(model_input))
    model = load_model(model_input, run_args.model_name, run_args.test_model_name)

    input_file = os.path.join(test_dir, f'{filename}.mat')
    logger.info(input_file)
    data, header_data = load_test_data(input_file)

    label, score, classes = run_classifier(data, header_data, model)

    results = defaultdict(list)
    results[filename]['filekey'] = filename
    for j, c in enumerate(classes):
        results[filename][c] = {"label": label[j], "score": score[j]}

    # results = {
    #     'filename': filename,
    #     **{'{}_label'.format(c): label[j] for j, c in enumerate(classes)},
    #     **{'{}_score'.format(c): score[j] for j, c in enumerate(classes)}
    # }

    results = json_serializer(results)

    with open(f'results_{filename}.json', 'w') as fp:
        json.dump(results, fp, indent=4)

    logger.info("Predictions saved!")

    return results