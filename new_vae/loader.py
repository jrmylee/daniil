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
    csv_dir = os.path.join(root_dir, csv_name)
    
    df = pd.read_csv(csv_dir)
    ds = get_dataset(df)
    
    train_ds, test_ds = get_train_test_set(ds)

    batch_size = 64
    train_steps = len(df) / batch_size
    
    return df, train_ds, test_ds

def get_dataset(csv, ds_dir=root_dir):
    audio_filenames, midi_filenames = [], []
    for index, row in csv.iterrows():
        audio, midi = os.path.join(ds_dir, row["audio_filename"]), os.path.join(ds_dir, row["midi_filename"])
        audio_filenames.append(audio)
        midi_filenames.append(midi)
    
    audio, midi = tf.constant(audio_filenames), tf.constant(midi_filenames)
    dataset = tf.data.Dataset.from_tensor_slices((audio, midi))    
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

def load_audio(audio_filepath, midi_filepath):
    print("loading audio")
    audio = tf.io.read_file(audio_filepath)
    audio, sample_rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=44100 * 2)
    audio = tfio.audio.resample(audio, 44100, 22050)
    audio = tf.reshape(audio, (22050 * 2, ))
    
    spec_clean, spec_dirty = get_feature(audio), [] # mel_spec(augment_audio(audio))
    paddings = tf.constant([[0, 3], [0,0]])
    spec_clean, spec_dirty = tf.pad(spec_clean, paddings, "CONSTANT"), [] #tf.pad(spec_dirty, paddings, "CONSTANT")
    spec_clean, spec_dirty = tf.expand_dims(spec_clean, -1), [] #tf.expand_dims(spec_dirty, -1)
    
    return spec_clean, audio_filepath

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
