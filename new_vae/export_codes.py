from tensorflow import keras
from tensorflow.keras import layers
from models.vqvae2 import *
from loader import get_dataset, split_data, load_audio
import numpy as np
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

model = VQVAETrainer(latent_dim=None, num_embeddings=None)
model.load_weights("./saved_models/savio_model_1")
model.compile(optimizer=keras.optimizers.Adam())


dataset = get_dataset()
dataset = dataset.map(load_audio, num_parallel_calls=AUTOTUNE)

save_dir = "/global/scratch/users/jrmylee/preprocessed/codes"

for element in dataset:
    spec, filepath = element

    filename = filepath.split("/")
    filename = filename[len(filename) - 1]

    print("saving: " + filename)

    save_path = os.path.join(save_dir, filename)

    top_index, bot_index = model.get_code_indices(spec)
    np.save(save_path, [top_index, bot_index])