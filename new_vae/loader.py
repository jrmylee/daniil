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


def mel_spec(spectrogram, hparams):
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=hparams.num_mel_bins,
        num_spectrogram_bins=int(hparams.n_fft/2)+1,
        sample_rate=22050,
        lower_edge_hertz=hparams.mel_lower_edge_hertz,
        upper_edge_hertz=hparams.mel_upper_edge_hertz,
        dtype=tf.dtypes.float32,
        name=None,
    )
    mel_f = mel_frequencies(
        n_mels=hparams.num_mel_bins + 2,
        fmin=hparams.mel_lower_edge_hertz,
        fmax=hparams.mel_upper_edge_hertz,
    )
    enorm = tf.dtypes.cast(
        tf.expand_dims(tf.constant(2.0 / (mel_f[2 : hparams.num_mel_bins + 2] - mel_f[:hparams.num_mel_bins])), 0),
        tf.float32)
    mel_matrix = tf.multiply(mel_matrix, enorm)
    mel_matrix = tf.divide(mel_matrix, tf.reduce_sum(mel_matrix, axis=0))
    mel_spectrogram = tf.tensordot(spectrogram,mel_matrix, 1)
    return mel_spectrogram

def inv_mel_spec(mel_spectrogram, hparams):
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=hparams.num_mel_bins,
        num_spectrogram_bins=int(hparams.n_fft/2)+1,
        sample_rate=22050,
        lower_edge_hertz=hparams.mel_lower_edge_hertz,
        upper_edge_hertz=hparams.mel_upper_edge_hertz,
        dtype=tf.dtypes.float32,
        name=None,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mel_inversion_matrix = tf.constant(
            np.nan_to_num(
                np.divide(mel_matrix.numpy().T, np.sum(mel_matrix.numpy(), axis=1))
            ).T
        )
    mel_spectrogram_inv = tf.tensordot(mel_spectrogram,tf.transpose(mel_inversion_matrix), 1)
    reconstructed_y_mel = inv_spectrogram_tensorflow(np.transpose(mel_spectrogram_inv), hparams)
    
    return reconstructed_y_mel

def get_feature(audio):
    spectrogram = spectrogram_tensorflow(audio, hparams)
    mel_spectrogram = mel_spec(spectrogram, hparams)
    return mel_spectrogram

def get_training_set():    
    ds = get_dataset()
  
    train_ds, test_ds = get_train_test_set(ds)

    batch_size = 64
    
    return train_ds, test_ds

def get_spectrogram_files(save_path):
    onlyfiles = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    return onlyfiles

def get_dataset(ds_dir=spectrogram_dir):
    files = get_spectrogram_files(ds_dir)
#     aug_files = [os.path.join(ds_dir, "aug", f) for f in files]
    files = [os.path.join(ds_dir, f) for f in files]
    
    
    files = tf.constant(files)
#     aug_files = tf.constant(aug_files)
    
    dataset = tf.data.Dataset.from_tensor_slices(files)    
    return dataset

def read_npy_file(item):
    stft_stacked = np.load(item.decode())
    real, imag = stft_stacked[:-1, :, :-1], stft_stacked[:-1, :, 1:]
    
    data = np.sqrt(real ** 2 + imag ** 2)
    data = np.pad(data, ((0,0), (0,1), (0,0)), 'constant')
    data = librosa.amplitude_to_db(data, ref=np.max)
    data = (-1 * data) / 80
    return data.astype(np.float32)

def load_audio(spec_filepath):
    print("loading spectrogram")
    transform_clean = tf.numpy_function(read_npy_file, [spec_filepath], [tf.float32])
#     spec_dirty = tf.numpy_function(read_npy_file, [dirty_spec_filepath], [tf.float32])
    
#     spec_dirty = tf.pad(spec_dirty, paddings, "CONSTANT")
#     spec_dirty = tf.reshape(spec_dirty, (176, 256, 1))
    
    return transform_clean

def get_train_test_set(ds, shuffle_buffer_size=1024, batch_size=64):
    test_ds = ds.take(200) 
    train_ds = ds.skip(200)

    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    train_ds = train_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.shuffle(buffer_size=shuffle_buffer_size)
    test_ds = test_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds
