from tensorflow import keras
from tensorflow.keras import layers
from models.vqvae2 import *
from models.pixelcnn import get_pixelcnn

top_input_size, bot_input_size = (128, 11, 1), (256, 22, 1)

vae = VQVAETrainer(latent_dim=None, num_embeddings=None)
top_pixelcnn, bot_pixelcnn = get_pixelcnn(top_input_size), get_pixelcnn(bot_input_size)

# Load Weights and Compile
vae.load_weights("./saved_models/savio_model_1")
vae.compile(optimizer=keras.optimizers.Adam())

top_pixelcnn.load_weights("./saved_models/pixel_model_1")
bot_pixelcnn.load_weights("./saved_models/pixel_model_bot")
top_pixelcnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="binary_crossentropy")
bot_pixelcnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="binary_crossentropy")


