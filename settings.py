backbone = 'vgg19' # f in paper
prototype_shape = (1000, 256, 1, 1)  # (num_prototypes, depth, height, width)
img_channels = 1 
num_classes = 10 # Nsynth num_classes because synth_lead class is removed 

# Specific prototypes for individual classes
class_specific = True

# Annotations csv
annotations_train = './data/nsynth-train/examples.csv' 
annotations_valid = './data/nsynth-valid/examples.csv' 
annotations_test = './data/nsynth-test/examples.csv' 
annotations_test_generated = './data/nsynth-generated/examples.csv' 
annotations_train_aug = './data/nsynth-train/examples.csv'

# Audio Dir
train_audio_dir = '/data01/DATASET/nsynth/nsynth-train/audio/' 
validation_audio_dir = '/data01/DATASET/nsynth/nsynth-valid/audio/' 
test_audio_dir = '/data01/DATASET/nsynth/nsynth-test/audio/' 
test_audio_generated_dir = '/data01/DATASET/nsynth/nsynth-generated/' 


experiment_run = 'nsynth-013'

# Learning rates and schedulers
joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-3

# Loss function coefficients
coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 25
num_warm_epochs = 5

# Early stopping 
patience = 5

push_start = 6
push_epochs = [i for i in range(num_train_epochs) if i % 3 == 0]
last_layer_convex_optimizations = 3

# Audio processing parameters
sample_rate = 16000
num_samples = 64000

# Spectrogram conversion parameters
n_fft = 2048
n_mels = 128
hop_length = 487


# Power spectrogram or dB units spect
power_or_db = 'd' # power spectrogram 'p', decibel dB units 'd'