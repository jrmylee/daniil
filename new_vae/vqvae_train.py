from models.vqvae2 import *
import tensorflow as tf
import time
from loader import get_training_set
import numpy as np
import datetime

train_dataset, test_dataset = get_training_set()
epochs = 20
batch_size=64
# set the dimensionality of the latent space to a plane for visualization later

# Model Definitions
vqvae_trainer = VQVAETrainer(latent_dim=None, num_embeddings=None)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5))

# Tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Checkpoint Model Saving
checkpoint_filepath = "/home/jerms/daniil/new_vae/saved_models/vqvae2_run_stft_3"
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
vqvae_trainer.fit(train_dataset, epochs=epochs, callbacks=callbacks)