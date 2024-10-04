"""
File: utils.py
Author: Koustav Mallick
Date: 04/10/2024

Description: [Add a brief description of the file here]
"""

import logging
from scipy.io import loadmat
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np, os, shutil, json
from tqdm import tqdm
import classifier

logger = logging.getLogger(__name__)


def load_and_run(datadict, output_folder, model):
    logger.info('Loading data from folder')
    for key, _ in datadict.items():
        folder_path = datadict[key]['path']
        if os.path.exists(folder_path):
            logger.info(f'Loading data from {folder_path}')
            matfiles = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

            for matfile in matfiles:
                data, header = load_data(folder_path, matfile)
                current_label, current_score, classes = classifier.run(data, header, model)

                # save_result(output_folder, matfile, current_label, current_score, classes)
        else:
            logger.info(f'Path {folder_path} does not exist')


def resample(input_signal, src_fs, tar_fs):
    dtype = input_signal.dtype
    audio_len = input_signal.shape[1]
    audio_time_max = 1.0 * (audio_len) / src_fs
    src_time = 1.0 * np.linspace(0, audio_len, audio_len) / src_fs
    tar_time = 1.0 * np.linspace(0, np.int(audio_time_max * tar_fs), np.int(audio_time_max * tar_fs)) / tar_fs
    for i in range(input_signal.shape[0]):
        if i == 0:
            output_signal = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
            output_signal = output_signal.reshape(1, len(output_signal))
        else:
            tmp = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
            tmp = tmp.reshape(1, len(tmp))
            output_signal = np.vstack((output_signal, tmp))
    return output_signal

def load_data(folder_path, filename):
    data_file = os.path.join(folder_path, filename)
    header_file = data_file.replace('.mat','.hea')
    x = loadmat(data_file)
    data = np.asarray(x['val'], dtype=np.float64)
    with open(header_file, 'r') as f:
        header = f.readlines()

    return data, header


def prepare_age_gender(age, gender):
    data = np.zeros(5,)
    if age >= 0:
        data[0] = age / 100
    if 'F' in gender:
        data[2] = 1
        data[4] = 1
    elif gender == 'Unknown':
        data[4] = 0
    elif 'f' in gender:
        data[2] = 1
        data[4] = 1
    else:
        data[3] = 1
        data[4] = 1
    return data


def kaggle_data_downloader(datadict, flag = False):
    api = KaggleApi()
    api.authenticate()
    if flag:
        logger.info('Starting data download process')
        if os.path.exists('input_data'):
            shutil.rmtree('input_data')
        os.makedirs('input_data')
        
        for key, value in tqdm(datadict.items(), desc='Downloading data'):
            print(key, value)
            if key not in os.listdir('input_data'):
                api.dataset_download_files(value['source'], path=f'input_data/{key}', unzip=True)
                folder_path = os.path.join('input_data', key, value['inside_folder'])
                for filename in os.listdir(folder_path):
                    os.rename(os.path.join(folder_path, filename), os.path.join('input_data', key, filename))
                shutil.rmtree(folder_path)
            datadict[key]['path'] = os.path.join('input_data', key)

        with open('datadict.json', 'w') as f:
            json.dump(datadict, f)
        
        logger.info('Finished downloading data')

def lsdir(rootdir="", suffix=".pth"):
    file_list = []
    assert os.path.exists(rootdir)
    for r, _, names in os.walk(rootdir):
        for name in names:
            if str(name).endswith(suffix):
                file_list.append(os.path.join(r, name))
    return file_list

def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')