import os
import tensorflow as tf
import numpy as np
import librosa
import pandas as pd
import random

from utils.echo import convolve_with_room, get_room_irs

irs = get_room_irs(100)

AUTOTUNE = tf.data.experimental.AUTOTUNE

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

def load_audio(audio_filepath):
    print("loading audio")

    audio_pairs = tf.numpy_function(read_audio, [audio_filepath], [tf.float32])

    return tf.squeeze(audio_pairs, axis=0)
