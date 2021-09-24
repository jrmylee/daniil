
import os
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class VAE:
  """
  VAE represents a Deep Convolutional autoencoder architecture
  with mirrored encoder and decoder components.
  """

  def __init__(self,
               input_shape, #shape of the input data
               conv_filters, #convolutional network filters
               conv_kernels, #convNet kernel size
               conv_strides, #convNet strides
               latent_space_dim):
    self.input_shape = input_shape # [28, 28, 1], in this case is 28 x 28 pixels on 1 channel for greyscale
    self.conv_filters = conv_filters # is a list for each layer, i.e. [2, 4, 8]
    self.conv_kernels = conv_kernels # list of kernels per layer, [1,2,3]
    self.conv_strides = conv_strides # stride for each filter [1, 2, 2], note: 2 means you are downsampling the data in half
    self.latent_space_dim = latent_space_dim # how many neurons on bottleneck
    self.reconstruction_loss_weight = 1000000

    self.encoder = None
    self.decoder = None
    self.model = None

    self._num_conv_layers = len(conv_filters)
    self._shape_before_bottleneck = None
    self._model_input = None

    self._build()

  def summary(self):
    self.encoder.summary()
    print("\n")
    self.decoder.summary()
    print("\n")
    self.model.summary()

  def _build(self):
    self._build_encoder()
    self._build_decoder()
    self._build_autoencoder()

  def compile(self, learning_rate=0.0001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    klDivergence = tf.keras.metrics.KLDivergence(name='kullback_leibler_divergence', dtype=None)
    
    self.model.compile(optimizer=optimizer, 
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[klDivergence])

  def train(self, x, x_hat, batch_size, num_epochs, train_steps, checkpoint_path):
    self.model.fit(x_hat, x, epochs=num_epochs, shuffle=True)

  def save(self, save_folder="."):
    print("Saving!")
    self._save_parameters(save_folder)
    self._save_weights(save_folder)

  def load_weights(self, weights_path):
    self.model.load_weights(weights_path)

  def reconstruct(self, spec):
      latent_representations = self.encoder.predict(spec)
      reconstructed_spec = self.decoder.predict(latent_representations)
      return reconstructed_spec, latent_representations
  
  def sample_from_latent_space(self, z):
      z_vector = self.decoder.predict(z)
      return z_vector

  @classmethod
  def load(cls, save_folder="."):
      parameters_path = os.path.join(save_folder, "parameters.pkl")
      with open(parameters_path, "rb") as f:
          parameters = pickle.load(f)
      autoencoder = VAE(*parameters)
      weights_path = os.path.join(save_folder, "weights.h5")
      autoencoder.load_weights(weights_path)
      return autoencoder

  def _save_parameters(self, save_folder):
      parameters = [
          self.input_shape,
          self.conv_filters,
          self.conv_kernels,
          self.conv_strides,
          self.latent_space_dim
      ]
      save_path = os.path.join(save_folder, "parameters.pkl")
      with open(save_path, "wb") as f:
          pickle.dump(parameters, f)

  def _save_weights(self, save_folder):
      save_path = os.path.join(save_folder, "weights.h5")
      self.model.save_weights(save_path)

#-----------AUTOENCODER----------#

  def _build_autoencoder(self):
    model_input = self._model_input
    model_output = self.decoder(self.encoder(model_input))
    self.model = Model(model_input, model_output, name="autoencoder")

#--------------DECODER------------#

  def _build_decoder(self):
    decoder_input = self._add_decoder_input()
    dense_layer = self._add_dense_layer(decoder_input)
    reshape_layer = self._add_reshape_layer(dense_layer)
    conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
    decoder_output = self._add_decoder_output(conv_transpose_layers)
    self.decoder = Model(decoder_input, decoder_output, name="decoder")

  def _add_decoder_input(self):
    return Input(shape=self.latent_space_dim, name="decoder_input")

  def _add_dense_layer(self, decoder_input):
    num_neurons = np.prod(self._shape_before_bottleneck) # [ 1, 2, 4] -> 8
    dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
    return dense_layer

  def _add_reshape_layer(self, dense_layer):
    return Reshape(self._shape_before_bottleneck)(dense_layer)

  def _add_conv_transpose_layers(self, x):
    """Add conv transpose blocks."""
    # Loop through all the conv layers in reverse order and
    # stop at the first layer
    for layer_index in reversed(range(1, self._num_conv_layers)):
      x = self._add_conv_transpose_layer(layer_index, x)
    return x

  def _add_conv_transpose_layer(self, layer_index, x):
    layer_num = self._num_conv_layers - layer_index
    conv_transpose_layer = Conv2DTranspose(
        filters=self.conv_filters[layer_index],
        kernel_size = self.conv_kernels[layer_index],
        strides = self.conv_strides[layer_index],
        padding = "same",
        name=f"decoder_conv_transpose_layer_{layer_num}"
    )
    x = conv_transpose_layer(x)
    x = ReLU(name=f"decoder_relu_{layer_num}")(x)
    x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
    return x

  def _add_decoder_output(self, x):
    conv_transpose_layer = Conv2DTranspose(
        filters = 1,
        kernel_size = self.conv_kernels[0],
        strides = self.conv_strides[0],
        padding = "same",
        name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
    )
    x = conv_transpose_layer(x)
    output_layer = Activation("sigmoid", name="sigmoid_output_layer")(x)
    return output_layer

#----------------ENCODER-----------------#

  def _build_encoder(self):
    encoder_input = self._add_encoder_input()
    conv_layers = self._add_conv_layers(encoder_input)
    bottleneck =  self._add_bottleneck(conv_layers)
    self._model_input = encoder_input
    self.encoder = Model(encoder_input, bottleneck, name="encoder")

  def _add_encoder_input(self):
    return Input(shape=self.input_shape, name="encoder_input")

  def _add_conv_layers(self, encoder_input):
    """Creates all convolutional blocks in encoder"""
    x = encoder_input
    for layer_index in range(self._num_conv_layers):
      x = self._add_conv_layer(layer_index, x)
    return x
  
  def _add_conv_layer(self, layer_index, x):
    """Adds a convolutional block to a graph of layers, consisting
    of Conv 2d + ReLu activation + batch normalization.
    """
    layer_number = layer_index + 1
    conv_layer = Conv2D(
        filters= self.conv_filters[layer_index],
        kernel_size = self.conv_kernels[layer_index],
        strides = self.conv_strides[layer_index],
        padding = "same",
        name = f"encoder_conv_layer_{layer_number}"
    )
    x = conv_layer(x)
    x = ReLU(name=f"encoder_relu_{layer_number}")(x)
    x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
    return x

#-------------Bottleneck (Latent Space)-------------#

  def _add_bottleneck(self, x):
    """Flatten data and add bottleneck with Gaussian sampling (Dense layer)"""
    self._shape_before_bottleneck = K.int_shape(x)[1:]
    x = Flatten()(x)
    self.mu = Dense(self.latent_space_dim,name="mu")(x)
    self.log_variance = Dense(self.latent_space_dim,
                              name="log_variance")(x)
    
    def sample_point_from_normal_distribution(args):
      mu, log_variance = args
      epsilon = K.random_normal(shape=K.shape(self.mu), mean=0., 
                                stddev=1.)
      sampled_point = mu + K.exp(log_variance / 2) * epsilon

      return sampled_point

    x = Lambda(sample_point_from_normal_distribution, 
               name="encoder_output")([self.mu, self.log_variance])
    return x
