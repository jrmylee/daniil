import tensorflow as tf
from tensorflow import keras

class QuantizerCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("Top Quantizer Embedding Count: ")
        print(self.model.vqvae.get_layer("vector_quantizer").embeddings_count, flush=True)
        print("Bottom Quantizer Embedding Count: ")
        print(self.model.vqvae.get_layer("vector_quantizer_1").embeddings_count, flush=True)
