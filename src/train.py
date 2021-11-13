from models.vqvae2 import *
import tensorflow as tf
from loader import get_dataset, split_data, load_audio
import datetime
import json
from types import SimpleNamespace
import os 

with open("config.json") as file:
    hparams = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    
    epochs = hparams.num_epochs
    batch_size= hparams.batch_size

    # for multi-gpu training
    mirrored_strategy = tf.distribute.MirroredStrategy()
    batch_size *= mirrored_strategy.num_replicas_in_sync

    dataset = get_dataset(ds_dir=os.path.join(hparams.dataset_dir, "original"),augmented_dir=os.path.join(hparams.dataset_dir, "echoed"))
    dataset = dataset.map(load_audio, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    with mirrored_strategy.scope():
        # Model Definitions
        vqvae_trainer = VQVAETrainer(latent_dim=None, num_embeddings=None)
        vqvae_trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=hparams.learning_rate))

        # Tensorboard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Checkpoint Model Saving
        checkpoint_filepath = os.path.join(hparams.model_save_dir, "macbook_model")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_best_only=True,
            monitor='loss')

        # Define keras callbacks
        callbacks = [
            tensorboard_callback,
            model_checkpoint_callback
        ]

        # Yeet
        vqvae_trainer.fit(dataset, epochs=epochs, callbacks=callbacks)

        vqvae_trainer.set_mode("reconstruction")

        vqvae_trainer.fit(dataset, epochs=epochs, callbacks=callbacks)