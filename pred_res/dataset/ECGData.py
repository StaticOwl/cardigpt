"""
File: ECGData.py
Project: potluck
Author: malli
Created: 06-10-2024
Description: This module contains the definition of ECGData class which is a PyTorch dataset for ECG data.
"""

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

__all__ = ['ECGData']


class ECGData(Dataset):

    def __init__(self, anno_pd, test=False, transform=None, data_dir=None):
        """
        Initialize ECGData instance.

        Parameters
        ----------
        anno_pd : pandas.DataFrame
            Annotation dataframe containing the following columns:
                * filename: string, name of the ECG file
                * fs: int, sampling frequency of the ECG file
                * age: int, age of the patient
                * gender: string, gender of the patient
        test : bool, optional
            Whether the dataset is for testing or not. If True, the labels are not loaded.
            Defaults to False.
        transform : callable, optional
            Transformation function to be applied to the ECG data.
            Defaults to None.
        data_dir : str, optional
            Directory where the ECG files are stored.
            Defaults to None.
        """
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
        """
        Get the ECG data and its corresponding label.

        Parameters
        ----------
        item : int
            Index of the ECG data.

        Returns
        -------
        data : torch.Tensor
            ECG data
        age_gender : torch.Tensor
            Age and gender of the patient
        label : torch.Tensor
            Label of the ECG data
        """
        if self.test:
            item_path = self.data[item]
            fs = self.fs[item]
            item = load_data(self.data_dir + item_path, src_fs=fs)
            item = self.transform(item)
            return item, item_path
        else:
            img_name = self.data[item]
            fs = self.fs[item]
            age = self.age[item]
            gender = self.gender[item]
            age_gender = prepare_data(age, gender)
            img = load_data(img_name, src_fs=fs)
            label = self.multi_labels[item]
            img = self.transform(img)
            return img, torch.from_numpy(age_gender).float(), torch.from_numpy(label).float()


def load_data(case, src_fs, tar_fs=257):
    """
    Load the ECG data from the given file and resample it to the target sampling frequency.

    Parameters
    ----------
    case : str
        Path to the ECG file
    src_fs : int
        Sampling frequency of the ECG file
    tar_fs : int, optional
        Target sampling frequency. Defaults to 257.

    Returns
    -------
    data : numpy.ndarray
        Resampled ECG data
    """
    x = loadmat(case)
    data = np.asarray(x['val'], dtype=np.float64)
    data = resample(data, src_fs, tar_fs)
    return data


def resample(input_signal, src_fs, tar_fs):
    """
    Resample the ECG data from the source sampling frequency to the target sampling frequency.

    Parameters
    ----------
    input_signal : numpy.ndarray
        ECG data
    src_fs : int
        Source sampling frequency
    tar_fs : int
        Target sampling frequency

    Returns
    -------
    output_signal : numpy.ndarray
        Resampled ECG data
    """
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
    """
    Prepare the age and gender data.

    Parameters
    ----------
    age : int
        Age of the patient
    gender : str
        Gender of the patient

    Returns
    -------
    data : numpy.ndarray
        Prepared age and gender data
    """
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
