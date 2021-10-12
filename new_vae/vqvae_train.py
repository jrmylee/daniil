from models.vqvae2 import *
import tensorflow as tf
import time
from loader import get_training_set
import numpy as np
import datetime

train_dataset, test_dataset = get_training_set()
epochs = 1
batch_size=64
# set the dimensionality of the latent space to a plane for visualization later

vqvae_trainer = VQVaeTrainer(latent_dim=256, num_embeddings=256)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5))

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

vqvae_trainer.fit(train_dataset, epochs=epochs, callbacks=[tensorboard_callback])

vqvae_trainer.save_weights("/home/jerms/daniil/new_vae/saved_models/vqvae_run_stft_6")
