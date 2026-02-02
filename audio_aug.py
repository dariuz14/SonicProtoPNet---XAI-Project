import torch
import random
import librosa

import torchaudio
import torchaudio.functional as F
from torchaudio.utils import download_asset

AUG_PER_SAMPLE = 1
AUGMENTATION_STRATEGY = 'snp' # 's' time stretch, 'n' background noise, 'pitch' pitch shift 
STRETCH_RATE_RANGE = (0.8, 0.9)
NOISE_DB_RANGE = (0, 10)
SEMITONES_SHIFT_RANGE = (1, 7)

AUG_PER_STRATEGY = AUG_PER_SAMPLE if AUG_PER_SAMPLE else round(AUG_PER_SAMPLE/len(AUGMENTATION_STRATEGY))

SAMPLE_NOISE_PATH = "tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav" 
NOISE, _ = torchaudio.load(download_asset(SAMPLE_NOISE_PATH))
NOISE = torch.cat((NOISE, NOISE, NOISE, NOISE, NOISE, NOISE, NOISE, NOISE), dim=1)

def augment_noise(signal):
    noise = NOISE[:, : signal.shape[1]] # reshaping noise len based on signal shape
    dB_bins = random.uniform(NOISE_DB_RANGE[0], NOISE_DB_RANGE[1])
    snr_dbs = torch.tensor([dB_bins], dtype=torch.float32)
    noisy_audio  = F.add_noise(signal, noise, snr_dbs)

    return noisy_audio[0:1]

def augment_pitch_shift(signal, sr):
    n_steps = random.uniform(SEMITONES_SHIFT_RANGE[0], SEMITONES_SHIFT_RANGE[1])
    signal_np = signal.numpy()[0]
    shifted = librosa.effects.pitch_shift(signal_np, sr=sr, n_steps=n_steps)
    return torch.tensor(shifted[None, :], dtype=torch.float32)

def augment_time_stretch(signal):
    rate = random.uniform(STRETCH_RATE_RANGE[0], STRETCH_RATE_RANGE[1])
    signal_np = signal.numpy()[0]
    stretched = librosa.effects.time_stretch(signal_np, rate=rate)
    return torch.tensor(stretched[None, :], dtype=torch.float32)
