import pandas as pd
import os
import tensorflow as tf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from preprocess import *
import numpy as np
import tensorflow_io as tfio
import librosa
from librosa import mel_frequencies

AUTOTUNE = tf.data.experimental.AUTOTUNE

root_dir = "/home/jerms/data/maestro-v3.0.0"
csv_name = "maestro-v3.0.0.csv"
_SEED = 2021

spectrogram_dir = "/home/jerms/disk1/stft_original"
augmented_dir = "/home/jerms/disk1/stft_augmented"

hparams = HParams(  
    # spectrogramming
    win_length = 2048,
    n_fft = 2048,
    hop_length= 256,
    ref_level_db = 50,
    min_level_db = -100,
    # mel scaling
    num_mel_bins = 256,
    mel_lower_edge_hertz = 0,
    mel_upper_edge_hertz = 10000,
    # inversion
    power = 1.5, # for spectral inversion
    griffin_lim_iters = 50,
    pad=True,
    #
)

def get_training_set():    
    ds = get_dataset()
    train_ds, test_ds = get_train_test_set(ds)    
    return train_ds, test_ds

def get_spectrogram_files(save_path):
    onlyfiles = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    return onlyfiles

def get_dataset(ds_dir=spectrogram_dir):
    
    files = get_spectrogram_files(ds_dir)
#     aug_files = [os.path.join(ds_dir, "aug", f) for f in files]
    files = [os.path.join(ds_dir, f) for f in files]
    
    # get augmented filenames
    aug_files = get_spectrogram_files(augmented_dir)
    files = [os.path.join(ds_dir, f) for f in aug_files] # get only good files that correspond to a augmented
    aug_files = [os.path.join(augmented_dir, f) for f in aug_files]
    
    files = tf.constant(files) # [path_to_file1... path to filen]
    aug_files = tf.constant(aug_files) # [path_to_badfile1 ... path to bad filen]
    
    dataset = tf.data.Dataset.from_tensor_slices((files, aug_files))    # => [[path to good file1 , path to badfile1], [], []]
    return dataset

def read_npy_file(item):
    stft_stacked = np.load(item.decode())
    real, imag = stft_stacked[:-1, :, :-1], stft_stacked[:-1, :, 1:]
    
    data = np.sqrt(real ** 2 + imag ** 2)
    data = np.pad(data, ((0,0), (0,1), (0,0)), 'constant')
    data = librosa.amplitude_to_db(data, ref=np.max)
    data = (-1 * data) / 80
    return data.astype(np.float32)

def load_audio(spec_filepath, dirty_spec_filepath):
    print("loading stft")
    transform_clean = tf.numpy_function(read_npy_file, [spec_filepath], [tf.float32])
    transform_clean = tf.squeeze(transform_clean, axis=0)
    spec_dirty = tf.numpy_function(read_npy_file, [dirty_spec_filepath], [tf.float32])   
    spec_dirty = tf.squeeze(spec_dirty, axis=0)
    return transform_clean, spec_dirty

def set_shapes(img, label, img_shape):
    img.set_shape(img_shape)
    label.set_shape(img_shape)
    return img, label

def get_train_test_set(ds, shuffle_buffer_size=1024, batch_size=64):
    test_ds = ds.take(200) 
    train_ds = ds.skip(200)
    
    img_shape = (1024, 88, 1)
    
    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    train_ds = train_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    
    test_ds = test_ds.shuffle(buffer_size=shuffle_buffer_size)
    test_ds = test_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda img, label: set_shapes(img, label, img_shape), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
   
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds
