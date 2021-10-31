from loader import get_dataset
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import numpy as np
from models.pixelcnn import PixelCNN

AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_codes(item):
    codes = np.load(item.decode())
    codes = codes.reshape(128, 11, 1)
    return codes.astype(np.float32)

def load_file(filepath):
    codes = tf.numpy_function(read_codes, [filepath], [tf.float32])
    return (codes,)

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

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

def image_preprocess(x):
  x['image'] = tf.cast(x['image'], tf.float32)
  return (x['image'],)  # (input, output) of the model

# Define a Pixel CNN network
dist = tfd.PixelCNN(
    image_shape=top_input_size,
    num_resnet=1,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=5,
    dropout_p=.3,
)

# Define the model input
image_input = tfkl.Input(shape=top_input_size)

# Define the log likelihood for the loss fn
log_prob = dist.log_prob(image_input)

# Define the model
model = tfk.Model(inputs=image_input, outputs=log_prob)
model.add_loss(-tf.reduce_mean(log_prob))

# Compile and train the model
model.compile(
    optimizer=tfk.optimizers.Adam(.001),
    metrics=[])

checkpoint_filepath = "/global/home/users/jrmylee/projects/daniil/new_vae/saved_models/pixel_model_1"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=True,
    monitor='loss')

callbacks = [
    model_checkpoint_callback
]

model.fit(train_ds, epochs=10, verbose=True, callbacks=callbacks)
