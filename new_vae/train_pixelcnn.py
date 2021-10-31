from loader import get_dataset
import tensorflow as tf
import numpy as np
from models.pixelcnn import PixelCNN

AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_codes(item):
    codes = np.load(item.decode())
    codes = codes.reshape(128, 11, 1)
    return codes.astype(np.float32)

def load_file(filepath):
    codes = tf.numpy_function(read_codes, [filepath], [tf.float32])
    return codes, filepath

def split_data(ds, shuffle_buffer_size=1024, batch_size=64):
    test_ds = ds.take(200) 
    train_ds = ds.skip(200)
        
    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    train_ds = train_ds.map(load_file, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    
    test_ds = test_ds.shuffle(buffer_size=shuffle_buffer_size)
    test_ds = test_ds.map(load_file, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
   
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds


dataset = get_dataset(ds_dir="/global/scratch/users/jrmylee/preprocessed/codes")
train_ds, test_ds = split_data(dataset)

top_input_size, bot_input_size = (128, 11), (256, 22)

top_model = PixelCNN(input_size=top_input_size, nb_channels=1)

top_model.fit(x=train_ds, validation_data=test_ds)