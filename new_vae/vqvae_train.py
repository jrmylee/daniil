from models.vqvae import *
import tensorflow as tf
import time
from loader import get_training_set
import numpy as np

train_dataset, test_dataset = get_training_set()
epochs = 10
# set the dimensionality of the latent space to a plane for visualization later

vqvae_trainer = VQVAETrainer(latent_dim=128, num_embeddings=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())

for epoch in range(1, epochs + 1):
    print("epoch: " + str(epoch))
    start_time = time.time()
    i = 0
    for train_x in train_dataset:
        i += 1
        vqvae_trainer.fit(train_x)
        if i == 5000:
            break
    end_time = time.time()

    print(end_time)

vqvae_trainer.save_weights("/home/jerms/daniil/new_vae/saved_models/vqvae_run_128_128")
