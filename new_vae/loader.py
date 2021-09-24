import pandas as pd
import os
import tensorflow as tf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from preprocess import *
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

root_dir = "/home/jerms/data/maestro-v3.0.0"
csv_name = "maestro-v3.0.0.csv"
_SEED = 2021

# this is hard coded for GTZan
# generates spectrograms from dataset
def generate_training_data(X_tr):
    print("generating training data")
    aspec = tospec(X_tr) 
    new_shape = ((0, 0), (0, 0), (0, 3), (0, 0))
    aspec = np.pad(aspec, pad_width=new_shape, mode='constant', constant_values=0)
    adata = split_spectrograms(aspec)
    return adata

def get_training_set():
    csv_dir = os.path.join(root_dir, csv_name)
    
    df = pd.read_csv(csv_dir)
    ds = get_dataset(df)
    
    train_ds = prepare_for_training(ds)

    batch_size = 64
    train_steps = len(df) / batch_size
    
    return df, train_ds

def get_dataset(csv, ds_dir=root_dir):
    audio_filenames, midi_filenames = [], []
    for index, row in csv.iterrows():
        audio, midi = row["audio_filename"], row["midi_filename"]
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
    return audio

def load_audio(audio_filepath, midi_filepath):
    audio = tf.io.read_file(audio_filepath)
    audio, sample_rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=44100)

    spec_clean, spec_dirty = prep(audio), prep(augment(audio))
    return spec_clean, spec_dirty

def prepare_for_training(ds, shuffle_buffer_size=1024, batch_size=64):
    # Randomly shuffle (file_path, label) dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Load and decode audio from file paths
    ds = ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    # Prepare batches
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds
