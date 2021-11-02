from models.vqvae2 import *
import tensorflow as tf
import time
from loader import get_dataset, split_data
import numpy as np
import datetime

dataset = get_dataset()
train_data, test_data = split_data(dataset)

epochs = 10
batch_size=64
# set the dimensionality of the latent space to a plane for visualization later

# Model Definitions
vqvae_trainer = VQVAETrainer(latent_dim=None, num_embeddings=None)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5))

# Tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Checkpoint Model Saving
checkpoint_filepath = "/global/home/users/jrmylee/projects/daniil/new_vae/saved_models/savio_unnorm"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=True,
    monitor='loss')

callbacks = [
    tensorboard_callback,
    model_checkpoint_callback
]

# Yeet
vqvae_trainer.fit(train_data, epochs=epochs, callbacks=callbacks)