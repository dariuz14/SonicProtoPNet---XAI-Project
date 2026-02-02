import os
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset

import numpy as np
import time

from helpers import makedir
from log import create_logger
from model import ProtoPNet

from audio_dataset import AudioDataset
import train_and_test as tnt

from settings import backbone, prototype_shape, num_classes, experiment_run
from settings import sample_rate, n_fft, hop_length, n_mels


model_dir = './saved_models/' + backbone + '/' + experiment_run + '/'
makedir(model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)

prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

mel_spectrogram_transformation = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)

cut_dimensions = (n_mels, n_mels)

img_size = n_mels

from settings import annotations_train, annotations_test, annotations_valid, annotations_test_generated
from settings import train_audio_dir, validation_audio_dir, test_audio_dir, test_audio_generated_dir, num_samples, power_or_db

# ---- LOADING DATASETS ----
train_dataset = AudioDataset(annotations_train, train_audio_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db, cut_dimensions)

np.random.seed(123) # previous 111
subset_size = len(train_dataset) // 3
subset_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
train_subset = Subset(train_dataset, subset_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4, pin_memory=False)

print(f'Numero di campioni nel subset training set: {len(train_subset)}')

validation_dataset = AudioDataset(annotations_valid, validation_audio_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db, cut_dimensions)
val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)

print(f'Numero di campioni nel validation set: {len(validation_dataset)}')

test_dataset = AudioDataset(annotations_test, test_audio_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db, cut_dimensions)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)

print(f'Numero di campioni nel test set: {len(test_dataset)}')

nsynth_test_generated = AudioDataset(annotations_test_generated, test_audio_generated_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db, cut_dimensions)
nsynth_test_generated_loader = DataLoader(nsynth_test_generated, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)

print(f'Numero di campioni nel test set sintetico: {len(nsynth_test_generated)}')

# Augmented dataset for push phase using "augmentations" parameter
train_dataset_aug = AudioDataset(annotations_train, train_audio_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db, cut_dimensions, augmentations=True)
train_subset_aug = Subset(train_dataset_aug, subset_indices)
train_loader_aug = DataLoader(train_subset_aug, batch_size=64, shuffle=True, pin_memory=False)

from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs, last_layer_convex_optimizations

from settings import coefs, class_specific

from settings import joint_optimizer_lrs, warm_optimizer_lrs, last_layer_optimizer_lr, joint_lr_step_size, patience

# Build Model
model = ProtoPNet(backbone, prototype_shape, num_classes, img_size, add_on_layers_type='bottleneck')
model = model.cuda()

# Hyperparameters initialization
joint_optimizer_params = [
    {'params': model.backbone.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
    {'params': model.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': model.prototype_layer, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_params)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

warm_optimizer_params = [
    {'params': model.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': model.prototype_layer, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_params)

last_layer_optimizer_lr = [{'params': model.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_lr)

# Training/ validation pipeline
log('start training')

start = time.time()

tnt.train_pipeline(model, train_loader, train_loader_aug, warm_optimizer, joint_optimizer, last_layer_optimizer,
            joint_lr_scheduler, num_train_epochs, num_warm_epochs, push_epochs, push_start,
            last_layer_convex_optimizations, class_specific, coefs,
            img_dir, prototype_img_filename_prefix, prototype_self_act_filename_prefix, proto_bound_boxes_filename_prefix, patience, model_dir=model_dir, validation=True, data_loader=val_loader, log=log) 

end = time.time()
log(f'Training completed in {(end-start)/60} minutes')

# Load best model for testing
model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))

# ---- Nsynth-test ----
final_accuracy_test, _ = tnt.test(model, test_loader, class_specific, log=log)

# ---- Synthetic-test ----
final_accuracy_test_generated, _ = tnt.test(model, nsynth_test_generated_loader, class_specific, log=log)

# ---- Prototypes analysis and evaluation ----

# Save plot in this path for nsynth-test
save_dir_nsynth_test = './plots/nsynth-test/'
tnt.evaluate_prototypes(model, test_loader, save_dir=save_dir_nsynth_test, log=log)

# Save plot in this path for nsynth-generated
save_dir_nsynth_generated = './plots/nsynth-generated/'
tnt.evaluate_prototypes(model, nsynth_test_generated_loader, save_dir=save_dir_nsynth_generated, log=log)

logclose()