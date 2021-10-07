from models.vqvae import *
import tensorflow as tf
import time
from loader import get_training_set
import numpy as np
import datetime

train_dataset, test_dataset = get_training_set()
epochs = 10
batch_size=64
# set the dimensionality of the latent space to a plane for visualization later

vqvae_trainer = VQVAETrainer(latent_dim=128, num_embeddings=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print("training")
vqvae_trainer.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback])

vqvae_trainer.save_weights("/home/jerms/daniil/new_vae/saved_models/vqvae_run_stft_2")
