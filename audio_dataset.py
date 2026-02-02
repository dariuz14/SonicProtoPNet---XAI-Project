import os
import pandas as pd
import torch
import librosa
import torchaudio
from torch.utils.data import Dataset
import random
from audio_aug import augment_noise, augment_pitch_shift, augment_time_stretch

AUGMENTATIONS = ["s", "n", "p"]

class AudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, target_sample_rate, num_samples, 
                 transformation, power_or_db, cut_dimensions=None, augmentations=False,):
       
        self.annotations = pd.read_csv(annotations_file, dtype={'target': 'string'})
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.transformation = transformation
        self.power_or_db = power_or_db
        self.cut_dimensions = cut_dimensions

        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal = self._get_signal_from_audio_path(audio_sample_path)
        return signal, label
    
    def _get_signal_from_audio_path(self, audio_sample_path):
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample(signal, sr)
        signal = self._right_pad_if_necessary(signal)

        if self.augmentations:
            aug = random.choice(AUGMENTATIONS)

            if aug == "n":
                signal = augment_noise(signal)
            elif aug == "p":
                signal = augment_pitch_shift(signal, sr)
            elif aug == "s":
                signal = augment_time_stretch(signal)

        signal = self.transformation(signal)

        if self.power_or_db == 'd':
            signal = librosa.power_to_db(signal)
        
        if self.cut_dimensions:
            signal = signal[:, :self.cut_dimensions[0], :self.cut_dimensions[1]]
        
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _get_audio_sample_path(self, index):
        filename = self.annotations.iloc[index, 0]
        return os.path.join(self.audio_dir, filename)

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1] # assuming 'target' is the second column

