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
import torch
from utils.metrics import cal_acc
from utils.predict_utils import load_model, load_test_data, run_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from utils.wanmetric import WandbLogger
from math import isnan
from datetime import datetime

date = datetime.now().strftime("%Y%m%d%H%M%S")

logger = logging.getLogger(__name__)
wandbLogger = WandbLogger(project_name='potluck', run_name=f'test_{date}')

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

    all_true_labels = []
    all_pred_logits = []

    for i, f in enumerate(input_files):
        logger.info("Predicting {} of {}".format(i + 1, num_files))
        tmp_input_file = os.path.join(test_dir, f)
        data, header_data = load_test_data(tmp_input_file)
        label, score, classes = run_classifier(data, header_data, model_all)
        if df_pred is None:
            columns = ['filename'] + ['{}_label'.format(c) for c in classes] + ['{}_score'.format(c) for c in classes]
            df_pred = pd.DataFrame(columns=columns)
        
        all_true_labels.append(label)
        all_pred_logits.append(score)

        row_data = {
            'filename': f,
            **{'{}_label'.format(c): label[j] for j, c in enumerate(classes)},
            **{'{}_score'.format(c): score[j] for j, c in enumerate(classes)}
        }
        df_tmp = pd.DataFrame([row_data])
        df_pred = pd.concat([df_pred, df_tmp], ignore_index=True)

    true_labels = torch.tensor(all_true_labels)
    pred_logits = torch.tensor(all_pred_logits)

    challenge_metric = cal_acc(true_labels, pred_logits)

    y_true = true_labels.numpy()
    y_prob = pred_logits.numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    y_true_flat = y_true.argmax(axis=1)
    y_pred_flat = y_pred.argmax(axis=1)

    logger.info("Calculating class distribution in the test dataset...")
    class_counts = y_true.sum(axis=0)
    total_samples = y_true.shape[0]
    class_distribution = pd.DataFrame({
        'Class': classes,
        'Count': class_counts,
        'Percentage': (class_counts / total_samples) * 100
    }).sort_values(by='Count', ascending=False)
    logger.info("\nClass Distribution in Test Dataset:")
    wandbLogger.log_table(name="Class Distribution", dataframe=class_distribution)

    for _, row in class_distribution.iterrows():
        logger.info(f"Class: {row['Class']} | Count: {int(row['Count'])} | Percentage: {row['Percentage']:.2f}%")

    most_common_count = class_counts.max()
    least_common_count = class_counts[class_counts > 0].min()  # Exclude classes with 0 count
    imbalance_ratio = most_common_count / least_common_count if least_common_count > 0 else float('inf')

    logger.info(f"Class Imbalance Ratio (Most Common:Least Common): {imbalance_ratio:.2f}")

    logger.info(f"Challenge Metric: {challenge_metric:.4f}")


    # holistic_metrics = {
    #     'challenge_metric': challenge_metric,
    #     'accuracy': accuracy_score(y_true_flat, y_pred_flat),
    #     'precision': precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
    #     'recall': recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
    #     'f1_score': f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
    #     'roc_auc': roc_auc_score(y_true, y_prob, average='macro', multi_class='ovo') if y_prob.ndim > 1 else None
    # }

    # for metric, score in holistic_metrics.items():
    #     logger.info(f"{metric.capitalize()}: {score:.4f}")

    class_specific_metrics = []
    for i, class_name in enumerate(classes):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        accuracy = accuracy_score(y_true_class, y_pred_class)
        precision = precision_score(y_true_class, y_pred_class, zero_division=0)
        recall = recall_score(y_true_class, y_pred_class, zero_division=0)
        f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true_class, y_pred[:, i])
        except ValueError:
            roc_auc = float('nan')
        roc_auc_str = f"{roc_auc:.4f}" if not isnan(roc_auc) else "NaN"
        logger.info(f"Class: {class_name} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | ROC AUC: {roc_auc_str}")
        class_specific_metrics.append({
            'Class': class_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC AUC': roc_auc
        })

    class_specific_metrics_df = pd.DataFrame(class_specific_metrics)
    wandbLogger.log_table(name="Class Specific Metrics", dataframe = class_specific_metrics_df)

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(len(classes))))
    wandbLogger.log_conf_matrix(y_true=y_true_flat, y_pred=y_pred_flat, class_name=list(range(len(classes))))
    logger.info("\nConfusion Matrix:")
    for i, row in enumerate(cm):
        row_str = " | ".join([f"{val:5d}" for val in row])
        logger.info(f"Class {classes[i]:15}: {row_str}")

    timestamp = time.time()
    wandbLogger.log_metric(name='Challenge Metric', value=challenge_metric)

    df_pred.to_csv(os.path.join(output_dir, f'predictions-{timestamp}.csv'), index=False)

    logger.info("Predictions saved!")
    wandbLogger.close()

    return {
        'class_specific_metrics': class_specific_metrics_df,
        'confusion_matrix': cm,
        'class_distribution': class_distribution
    }