from loader import get_dataset
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from models.pixel_cnn import get_pixelcnn

AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_codes(item):
    codes = np.load(item.decode(), allow_pickle=True)
    codes = codes[1].reshape(128, 11, 1)
    data = np.where(data < (0.33 * 512), 0, 1)
    return codes.astype(np.float32)

def load_file(filepath):
    codes = tf.numpy_function(read_codes, [filepath], [tf.float32])
    codes = tf.squeeze(codes, axis=0)
    return (codes, codes)

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

top_input_size, bot_input_size = (128, 11, 1), (256, 22, 1)

# top_pixel_cnn = get_pixelcnn(top_input_size)
bot_pixel_cnn = get_pixelcnn(bot_input_size)

adam = keras.optimizers.Adam(learning_rate=0.0005)
bot_pixel_cnn.compile(optimizer=adam, loss="binary_crossentropy")

bot_pixel_cnn.summary()

checkpoint_filepath = "/global/home/users/jrmylee/projects/daniil/new_vae/saved_models/pixel_model_bot"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=True,
    monitor='loss')

callbacks = [
    model_checkpoint_callback
]

bot_pixel_cnn.fit(
    x=train_ds,  epochs=10, callbacks=callbacks
)

