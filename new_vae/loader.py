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

spectrogram_dir = "/home/jerms/disk1/spectrograms"

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
    print(mel_matrix.shape)
    mel_f = mel_frequencies(
        n_mels=hparams.num_mel_bins + 2,
        fmin=hparams.mel_lower_edge_hertz,
        fmax=hparams.mel_upper_edge_hertz,
    )
    print(mel_f.shape)
    enorm = tf.dtypes.cast(
        tf.expand_dims(tf.constant(2.0 / (mel_f[2 : hparams.num_mel_bins + 2] - mel_f[:hparams.num_mel_bins])), 0),
        tf.float32)
    print(enorm.shape)
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
    onlyfiles = [os.path.join(save_path,f) for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    return onlyfiles

def get_dataset(ds_dir=spectrogram_dir):
    files = get_spectrogram_files(ds_dir)
    
    specs = tf.constant(files)
    dataset = tf.data.Dataset.from_tensor_slices(specs)    
    return dataset

def augment_audio(audio):
    sr = 22050
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.15, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])
    return augment(audio)

def read_npy_file(item):
    data = np.load(item.decode())
    return data.astype(np.float32)

def load_audio(audio_filepath):
    print("loading spectrogram")
    spec_clean = tf.numpy_function(read_npy_file, [audio_filepath], [tf.float32])
    paddings = tf.constant([[0,0], [0, 3], [0,0]])
    spec_clean, spec_dirty = tf.pad(spec_clean, paddings, "CONSTANT"), [] #tf.pad(spec_dirty, paddings, "CONSTANT")
    shape = spec_clean.shape
    if(hasattr(shape, 'numpy')):
        print("has shape")
    else:
        print("not shape")

    spec_clean = tf.reshape(spec_clean, (176, 256, 1))
    return spec_clean

def get_train_test_set(ds, shuffle_buffer_size=1024, batch_size=64):
    test_ds = ds.take(200) 
    train_ds = ds.skip(200)

    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    train_ds = train_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.shuffle(buffer_size=shuffle_buffer_size)
    test_ds = test_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
            
    return train_ds, test_ds
