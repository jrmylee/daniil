from models.vqvae2 import *
import tensorflow as tf
from loader import get_dataset, split_data
import datetime

mirrored_strategy = tf.distribute.MirroredStrategy()
batch_size = 32
batch_size *= mirrored_strategy.num_replicas_in_sync

dataset = get_dataset()
train_data, test_data = split_data(dataset, batch_size=batch_size)

epochs = 30
# set the dimensionality of the latent space to a plane for visualization later

with mirrored_strategy.scope():
    # Model Definitions
    vqvae_trainer = VQVAETrainer(latent_dim=None, num_embeddings=None)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

    # Tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint_filepath = "/global/home/users/jrmylee/projects/daniil/new_vae/saved_models/echoed_model_2"
    
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
