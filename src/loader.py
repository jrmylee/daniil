import os
import tensorflow as tf
import numpy as np
import librosa

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
    db = mask_stft(db)
    stft = -1 * db / 80.
    stft = stft[:-1, :-1].reshape(1024, 88, 1)
    return stft.astype(np.float32)

def load_audio(spec_filepath, dirty_spec_filepath):
# def load_audio(spec_filepath):
    print("loading stft")

    transform_clean = tf.numpy_function(read_stft_file, [spec_filepath], [tf.float32])
    transform_clean = tf.squeeze(transform_clean, axis=0)
    transform_dirty = tf.numpy_function(read_reverb_file, [dirty_spec_filepath], [tf.float32])   
    transform_dirty = tf.squeeze(transform_dirty, axis=0)
    return transform_clean, transform_dirty
