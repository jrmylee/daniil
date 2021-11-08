import os
import tensorflow as tf
import numpy as np
import librosa

AUTOTUNE = tf.data.experimental.AUTOTUNE

spectrogram_dir = "/Users/llewyn/Documents/data/stft/original"
augmented_dir = "/Users/llewyn/Documents/data/stft/echoed"

def get_dataset(ds_dir=spectrogram_dir):
    files = [f for f in os.listdir(ds_dir) if os.path.isfile(os.path.join(ds_dir, f))]
    files = [os.path.join(ds_dir, f) for f in files]
    
    # get augmented filenames
    aug_files = [f for f in os.listdir(augmented_dir) if os.path.isfile(os.path.join(augmented_dir, f))]
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
   
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds


# reads a file storing the result of an stft of shape (freq, time, 2), where the last channel is real, imag pairs
# Returns the magnitude of the spectrogram, normalized
def read_stft_file(item):
    stft = np.load(item.decode())
    stft = -1 * librosa.amplitude_to_db(stft, ref=np.max) / 80.
    stft = stft[:-1, :-1].reshape(1024, 88, 1)
    return stft.astype(np.float32)

def normalize_mel(x):
    m_a = 0.0661371661726
    m_b = 0.113718730221
    p_a = 0.8
    p_b = 0.
    _a = np.asarray([m_a, p_a])[None, None, None, :]
    _b = np.asarray([m_b, p_b])[None, None, None, :]

    normalized_mel = tf.clip_by_value(_a * x + _b, -1.0, 1.0)
    return normalized_mel

def denormalize_mel(x):
    m_a = 0.0661371661726
    m_b = 0.113718730221
    p_a = 0.8
    p_b = 0.
    _a = np.asarray([m_a, p_a])[None, None, None, :]
    _b = np.asarray([m_b, p_b])[None, None, None, :]

    return (x - _b) / _a

# reads a file storing the result of a melspectrogram
# returns the melspectrogram, normalized by pre-determined factors in GanSynth
def read_mel_file(item):
    mel = np.load(item.decode())
    normalized_mel = normalize_mel(mel)

    return normalized_mel

def load_audio(spec_filepath, dirty_spec_filepath):
# def load_audio(spec_filepath):
    print("loading stft")

    transform_clean = tf.numpy_function(read_stft_file, [spec_filepath], [tf.float32])
    transform_clean = tf.squeeze(transform_clean, axis=0)
    transform_dirty = tf.numpy_function(read_stft_file, [dirty_spec_filepath], [tf.float32])   
    transform_dirty = tf.squeeze(transform_dirty, axis=0)
    return transform_clean, transform_dirty
