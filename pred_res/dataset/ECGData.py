"""
File: ECGData.py
Project: potluck
Author: malli
Created: 06-10-2024
Description: write_a_description
"""
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class ECGData(Dataset):

    def __init__(self, anno_pd, test=False, transform=None, data_dir=None):
        self.anno_pd = anno_pd
        self.test = test
        self.transform = transform
        self.data_dir = data_dir

        if self.test:
            self.data = anno_pd['filename'].tolist()
            self.fs = anno_pd['fs'].tolist()
        else:
            self.data = anno_pd['filename'].tolist()
            labels = anno_pd.iloc[:, 4:].values
            self.multi_labels = [labels[i, :] for i in range(labels.shape[0])]
            self.age = anno_pd['age'].tolist()
            self.gender = anno_pd['gender'].tolist()
            self.fs = anno_pd['fs'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.test:
            item_path = self.data[item]
            fs = self.fs[item]
            item = load_data(self.data_dir + item_path, src_fs=fs)
            item = self.transform(item)
            return item, item_path
        else:
            item_name = self.data[item]
            fs = self.fs[item]
            age = self.age[item]
            gender = self.gender[item]
            age_gender = prepare_data(age, gender)
            item = load_data(item_name, src_fs=fs)
            label = self.multi_labels[item]
            item = self.transform(item)
            return item, torch.from_numpy(age_gender).float(), torch.from_numpy(label).float()


def load_data(case, src_fs, tar_fs=257):
    x = loadmat(case)
    data = np.asarray(x['val'], dtype=np.float64)
    data = resample(data, src_fs, tar_fs)
    return data


def resample(input_signal, src_fs, tar_fs):
    global output_signal
    if src_fs != tar_fs:
        dtype = input_signal.dtype
        audio_len = input_signal.shape[1]
        audio_time_max = 1.0 * (audio_len) / src_fs
        src_time = 1.0 * np.linspace(0, audio_len, audio_len) / src_fs
        tar_time = 1.0 * np.linspace(0, int(audio_time_max * tar_fs), int(audio_time_max * tar_fs)) / tar_fs
        for i in range(input_signal.shape[0]):
            if i == 0:
                output_signal = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
                output_signal = output_signal.reshape(1, len(output_signal))
            else:
                tmp = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
                tmp = tmp.reshape(1, len(tmp))
                output_signal = np.vstack((output_signal, tmp))
    else:
        output_signal = input_signal
    return output_signal


def prepare_data(age, gender):
    data = np.zeros(5, )
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
