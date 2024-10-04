"""
File: run_classifier.py
Author: Koustav Mallick
Date: 04/10/2024

Description: [Add a brief description of the file here]
"""

import torch
import logging
import numpy as np
from utils import prepare_age_gender, resample, lsdir
import math
import shutil
import pandas as pd
import model_repo

logger = logging.getLogger(__name__)
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_count = torch.cuda.device_count()
else:
    logger.warning('CUDA is not available. Using CPU instead.')
    device = torch.device('cpu')


def run(data, header, model):
    weight_list = ['./magic_weight0.npz', './magic_weight1.npz', './magic_weight2.npz',
                   './magic_weight3.npz', './magic_weight4.npz']
    num_classes = 24
    target_fs = 250
    src_fs = int(header[0].split(' ')[2].strip())
    m_header = process_header(header)
    m_header = m_header.to(device)

    win_length = 4096

    sig = torch.nn.Sigmoid()
    data = process_data(data, src_fs, target_fs, win_length)
    inputs = data.to(device)
    val_length = inputs.shape[2]
    overlap = 256
    patch_number = math.ceil(abs(val_length - win_length) / (win_length - overlap)) + 1
    if patch_number > 1:
        start = int((val_length - win_length) / (patch_number - 1))
    score = 0
    combined_label = 0

    for j in range(len(model)):
        model_one = model[j]
        for i in range(patch_number):
            if i == 0:
                logit = model_one(inputs[:, :, 0: val_length], m_header)
                logits_prob = sig(logit)
            elif i == patch_number - 1:
                logit = model_one(inputs[:, :, val_length - win_length: val_length], m_header)
                logits_prob_tmp = sig(logit)
                logits_prob = (logits_prob + logits_prob_tmp) / patch_number
            else:
                logit = model_one(inputs[:, :, i * start:i * start + win_length], m_header)
                logits_prob_tmp = sig(logit)
                logits_prob = logits_prob + logits_prob_tmp

        # using the threshold to check each model
        A = np.load(weight_list[j])
        threshold = A['arr_0']
        score_tmp, pred_label = output_label(logits_prob, threshold, num_classes)

        # the label
        combined_label = combined_label + pred_label

        # The probability
        score = score + score_tmp

    score = score / len(model)
    combined_label = combined_label / len(model)
    max_index = np.argmax(combined_label, 1)
    combined_label[0, max_index] = 1
    threshold_tmp = 0.5
    combined_label[combined_label >= threshold_tmp] = 1
    combined_label[combined_label < threshold_tmp] = 0


    current_label = np.squeeze(combined_label.astype(np.int))
    current_score = np.squeeze(score)

    # Get the label
    label_file_dir = './utils/dx_mapping_scored.csv'
    label_file = pd.read_csv(label_file_dir)
    equivalent_classes = ['59118001', '63593006', '17338001']
    classes = sorted(list(set([str(name) for name in label_file['SNOMED CT Code']]) - set(equivalent_classes)))

    return current_label, current_score, classes

def load(model_input):
    model_list = ['./load_model/48-0.6740-split0.pth',
                  './load_model/42-0.6701-split1.pth',
                  './load_model/40-0.6777-split2.pth',
                  './load_model/42-0.6749-split3.pth',
                  './load_model/47-0.6791-split4.pth']
    for i in range(5):
        shutil.copy(model_list[i], model_input)
    model_list = lsdir(rootdir=model_input, suffix=".pth")
    split_list = ['split0', 'split1', 'split2', 'split3', 'split4']
    resumes = []
    for split in split_list:
        sub_list = [i for i in model_list if split in i]
        accuracy = np.array([float(i.split('-')[-2]) for i in sub_list])
        resumes.append(sub_list[int(np.argmax(accuracy))])

    model_all = []

    for resume in resumes:
        model = getattr(model_repo, 'resnet')(in_channel=12, out_channel=24)
        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            if device_count > 1:
                model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(resume))
        else:
            model.load_state_dict(torch.load(resume, map_location=device))

        model.to(device)
        model.eval()
        model_all.append(model)

    return model_all

def output_label(logits_prob, threshold, num_classes):
    pred_label = np.zeros(num_classes, dtype=int)
    _, y_pre_label = torch.max(logits_prob, 1)
    y_pre_label = y_pre_label.cpu().detach().numpy()
    pred_label[y_pre_label] = 1

    score_tmp = logits_prob.cpu().detach().numpy()

    y_pre = (score_tmp - threshold) >= 0
    pred_label = pred_label + y_pre
    pred_label[pred_label > 1.1] = 1
    return score_tmp, pred_label

def process_data(data, src_fs, target_fs, win_length):
    data = resample(data, src_fs, target_fs)
    num = data.shape[1]
    if num < win_length:
        zeros_padding = np.zeros(shape=(data.shape[0], win_length - num), dtype=np.float32)
        data = np.hstack((data, zeros_padding))
    data = data.astype(np.float32)
    data = torch.from_numpy(data)
    data = torch.unsqueeze(data, 0)

    return data

def process_header(header):
    for lines in header:
        if lines.startswith('#Age'):
            tmp = lines.split(': ')[1].strip()
            if tmp == 'NaN':
                age = -1
            else:
                age = int(tmp)
        if lines.startswith('#Sex'):
            tmp = lines.split(': ')[1].strip()
            if tmp == 'NaN':
                gender = 'Unknown'
            else:
                gender = tmp

    data = prepare_age_gender(age, gender)
    data = torch.from_numpy(data).float()
    data = torch.unsqueeze(data, 0)
    return data