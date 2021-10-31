from loader import get_dataset, split_data
import tensorflow as tf
import numpy as np
from models.pixelcnn import PixelCNN

dataset = get_dataset(ds_dir="/global/scratch/users/jrmylee/preprocessed/codes")
train_ds, test_ds = split_data(dataset)

top_input_size, bot_input_size = (128, 11), (256, 22)

top_model = PixelCNN(input_size=top_input_size, nb_channels=1)

top_model.fit(x=train_ds, validation_data=test_ds)