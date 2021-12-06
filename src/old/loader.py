import os
import tensorflow as tf
import numpy as np
import librosa
import pandas as pd
import random

from utils.echo import convolve_with_room, get_room_irs

irs = get_room_irs()

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_dataset(ds_dir, augmented_dir):
    # get augmented filenames
    aug_files = [f for f in os.listdir(augmented_dir) if os.path.isfile(os.path.join(augmented_dir, f))]
    files = aug_files
    
    files = [os.path.join(ds_dir, f) for f in files]
    aug_files = [os.path.join(augmented_dir, f) for f in aug_files]
    
    files = tf.constant(files) # [path_to_file1... path to filen]
    aug_files = tf.constant(aug_files) # [path_to_badfile1 ... path to bad filen]
    
    dataset = tf.data.Dataset.from_tensor_slices((files, aug_files))    # => [[path to good file1 , path to badfile1], [], []]
    return dataset

def get_audio_dataset(ds_path, mapping_filename):
    csv_path = os.path.join(ds_path, mapping_filename)
    csv = pd.read_csv(csv_path)
    files = []
    for index, row in csv.iterrows():
        full_audio_path = os.path.join(ds_path, row["audio_filename"])

        files.append(full_audio_path)
    
    dataset = tf.data.Dataset.from_tensor_slices(files)
    return dataset

def split_audio_dataset(ds, shuffle_buffer_size=1024, batch_size=64):
    test_ds = ds.take(200) 
    train_ds = ds.skip(200)
        
    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    train_ds = train_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.unbatch()
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    
    test_ds = test_ds.shuffle(buffer_size=shuffle_buffer_size)
    test_ds = test_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.unbatch()
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
   
    train_ds = train_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    return train_ds, test_ds

# Splits input dataset into training and test sets
# to be used with get_dataset()
def split_data(ds, shuffle_buffer_size=1024, batch_size=64):
    test_ds = ds.take(200) 
    train_ds = ds.skip(200)
        
    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    train_ds = train_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    
    test_ds = test_ds.shuffle(buffer_size=shuffle_buffer_size)
    test_ds = test_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
   
    train_ds = train_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    return train_ds, test_ds

def mask_stft(spec, mask_ratio=0.5):
    total_length = spec.shape[0] * spec.shape[1]
    mask = np.zeros(total_length, dtype=int)
    mask[:int(total_length * mask_ratio)] = 1
    np.random.shuffle(mask)
    mask = mask.astype(bool).reshape(spec.shape)
    spec[mask == 1] = -80.
    return spec

def read_audio(item):
    def chunk(x):
        chunk_length = 2048 * 22
        if len(x) % chunk_length != 0:
            multiple = np.ceil(len(x) / chunk_length)
            pad_amount = chunk_length * multiple - len(x)
            x = np.pad(x, (0, int(pad_amount)), 'constant', constant_values=(0, 0))

        # split in 2 second chunks and export to files 
        arr = []
        for i in range(0, len(x), 2048 * 22):
            y = x[i : i + 2048 * 22]
            arr.append(y)
        return np.array(arr)
    x, sr = librosa.load(item, sr=44100, mono=True)
    y = convolve_with_room(x, random.choice(irs))[:len(x)]
    
    x_chunks, y_chunks = chunk(x), chunk(y)
    data = np.stack([x_chunks, y_chunks], axis=2)
        
    return data.astype(np.float32)

# reads a file storing the result of an stft of shape (freq, time, 2), where the last channel is real, imag pairs
# Returns the magnitude of the spectrogram, normalized
def read_stft_file(item):
    stft = np.load(item.decode())
    stft = -1 * librosa.amplitude_to_db(stft, ref=np.max) / 80.
    stft = stft[:-1, :-1].reshape(1024, 88, 1)
    return stft.astype(np.float32)

def read_reverb_file(item):
    stft = np.load(item.decode())
    db = librosa.amplitude_to_db(stft, ref=np.max)
    # db = mask_stft(db)
    stft = -1 * db / 80.
    stft = stft[:-1, :-1].reshape(1024, 88, 1)
    return stft.astype(np.float32)

def load_stft(spec_filepath, dirty_spec_filepath):
# def load_audio(spec_filepath):
    print("loading stft")

    transform_clean = tf.numpy_function(read_stft_file, [spec_filepath], [tf.float32])
    transform_clean = tf.squeeze(transform_clean, axis=0)
    transform_dirty = tf.numpy_function(read_reverb_file, [dirty_spec_filepath], [tf.float32])   
    transform_dirty = tf.squeeze(transform_dirty, axis=0)
    return transform_clean, transform_dirty

def load_audio(audio_filepath):
    print("loading audio")

    audio_pairs = tf.numpy_function(read_audio, [audio_filepath], [tf.float32])

    return tf.squeeze(audio_pairs, axis=0)
