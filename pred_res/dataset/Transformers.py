"""
File: Transformers.py
Project: potluck
Author: malli
Created: 06-10-2024
Description: Data transformation functions for ECG dataset
"""

import random
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        """
        Composes multiple transforms together.
        
        Args:
            transforms (list): List of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, seq):
        """
        Call method to apply transforms in sequence.
        
        Args:
            seq (numpy.ndarray): Input sequence.
        
        Returns:
            numpy.ndarray: Transformed sequence.
        """
        for t in self.transforms:
            seq = t(seq)
        return seq

class RandomClip(object):
    def __init__(self, len=72000):
        """
        Initializes RandomClip transform.
        
        Args:
            len (int): Length of the clipped sequence.
        """
        self.len = len

    def __call__(self, seq):
        """
        Clips the sequence randomly.
        
        Args:
            seq (numpy.ndarray): Input sequence.
        
        Returns:
            numpy.ndarray: Clipped sequence.
        """
        if seq.shape[1] >= self.len:
            start = random.randint(0, seq.shape[1] - self.len)
            seq = seq[:, start:start + self.len]
        else:
            left = random.randint(0, self.len - seq.shape[1])
            right = self.len - seq.shape[1] - left
            zeros_padding1 = np.zeros(shape=(seq.shape[0], left), dtype=np.float32)
            zeros_padding2 = np.zeros(shape=(seq.shape[0], right), dtype=np.float32)
            seq = np.hstack((zeros_padding1, seq, zeros_padding2))
        return seq

class Normalize(object):
    def __init__(self, type="0-1"):
        """
        Initializes Normalize transform.
        
        Args:
            type (str): Type of normalization.
        """
        self.type = type

    def __call__(self, seq):
        """
        Normalizes the sequence based on the given type.
        
        Args:
            seq (numpy.ndarray): Input sequence.
        
        Returns:
            numpy.ndarray: Normalized sequence.
        """
        if self.type == "0-1":
            for i in range(seq.shape[0]):
                if np.sum(seq[i, :]) == 0:
                    seq[i, :] = seq[i, :]
                else:
                    seq[i, :] = (seq[i, :] - seq[i, :].min()) / (seq[i, :].max() - seq[i, :].min())
        elif self.type == "mean-std":
            for i in range(seq.shape[0]):
                if np.sum(seq[i, :]) == 0:
                    seq[i, :] = seq[i, :]
                else:
                    seq[i, :] = (seq[i, :] - seq[i, :].mean()) / seq[i, :].std()
        elif self.type == "none":
            seq = seq
        else:
            raise NameError('This normalization is not included!')
        return seq

class Retype(object):
    def __call__(self, seq):
        """
        Converts the sequence to float32 type.
        
        Args:
            seq (numpy.ndarray): Input sequence.
        
        Returns:
            numpy.ndarray: Sequence with float32 data type.
        """
        return seq.astype(np.float32)

class ValClip(object):
    def __init__(self, len=72000):
        """
        Initializes ValClip transform.
        
        Args:
            len (int): Length of the clipped sequence.
        """
        self.len = len

    def __call__(self, seq):
        """
        Clips the sequence to a fixed length.
        
        Args:
            seq (numpy.ndarray): Input sequence.
        
        Returns:
            numpy.ndarray: Clipped sequence.
        """
        if seq.shape[1] >= self.len:
            seq = seq
        else:
            zeros_padding = np.zeros(shape=(seq.shape[0], self.len - seq.shape[1]), dtype=np.float32)
            seq = np.hstack((seq, zeros_padding))
        return seq
    
class AddNoise(object):
    def __init__(self, noise_factor=0.01):
        self.noise_factor = noise_factor

    def __call__(self, seq):
        noise = np.random.normal(0, self.noise_factor, seq.shape)
        return seq + noise
    
class TimeWarp(object):
    def __init__(self, stretch_factor=0.2):
        self.stretch_factor = stretch_factor

    def __call__(self, seq):
        factor = 1 + random.uniform(-self.stretch_factor, self.stretch_factor)
        warped_seq = np.zeros((seq.shape[0], int(seq.shape[1] * factor)))
        for i in range(seq.shape[0]):
            warped_seq[i] = np.interp(
                np.linspace(0, len(seq[i]), int(len(seq[i]) * factor)),
                np.arange(len(seq[i])), 
                seq[i]
            )

        return warped_seq
    
class MagnitudeScaling(object):
    def __init__(self, scale_factor=0.1):
        self.scale_factor = scale_factor

    def __call__(self, seq):
        factor = 1 + random.uniform(-self.scale_factor, self.scale_factor)
        return seq * factor