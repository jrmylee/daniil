# Configuration

import json
config_file = open('config.json',)
config = json.load(config_file)
training_params = config["training_params"]

# Training Parameters
LEARNING_RATE = training_params['learning_rate']
EPOCHS = training_params['num_epochs']
BATCH_SIZE = training_params['batch_size']
VECTOR_DIM = training_params['vector_dimension']

# Pre-Processing Section
HOP_SIZE = training_params['hop_size']
SAMPLE_RATE = training_params['sr']
MIN_LEVEL_DB = training_params['min_level_db']
REF_LEVEL_DB = training_params['ref_level_db']
TIME_AXIS_LENGTH = training_params['time_axis_length']
spec_split = 1

from preprocess import *
from augment import augment_audio
from model import CVAE, train_step
from loader import get_training_set


def load_model(checkpoint):
  vae = VAE.load(checkpoint)
  vae.summary()
  vae.compile(LEARNING_RATE)
  return vae
# --------------------------------------------------------------------------------

df, ds = get_training_set()

training_run_name = "my_melspecvae_model"
checkpoint_save_directory = "./saved_models/"

current_time = get_time_stamp()

train_steps = len(df) / BATCH_SIZE

vae = VAE(
      input_shape = (TIME_AXIS_LENGTH, HOP_SIZE, 1),
      conv_filters=(512, 256, 128, 64, 32),
      conv_kernels=(3, 3, 3, 3, 3),
      conv_strides=(2, 2, 2, 2, (2,1)),
      latent_space_dim = VECTOR_DIM
  )
vae.summary()
vae.compile(learning_rate)

i = 0
for batch in ds:
  print(str(i), " iteration!")
  x, x_hat = batch
  vae.train(x, x_hat, 64, epochs, train_steps, chkpt_pth)

vae.save(f"{checkpoint_save_directory}{training_run_name}_{current_time}_h{HOP_SIZE}_w{TIME_AXIS_LENGTH}_z{VECTOR_DIM}")