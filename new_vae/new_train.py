import tensorflow as tf
import time
from models.cvae import CVAE
from loader import get_training_set
import numpy as np

_, train_dataset, test_dataset = get_training_set()
batch_size = 32
epochs = 10
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 16

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

# @tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    print("training step")
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)
optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, metrics=["mae"])

for test_batch in test_dataset.take(1):
    test_good_sample = test_batch[0][0:num_examples_to_generate, :, :, :]
    test_good_label = test_batch[1][0:16]
    
for epoch in range(1, epochs + 1):
    print("epoch: " + str(epoch))
    start_time = time.time()
    for train_x in train_dataset:
        good_x, bad_x = train_x
        model.fit(good_x, batch_size=16)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        good_x, bad_x = test_x
        loss(model.compute_loss(good_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))

model.save("/home/jerms/daniil/new_vae/saved_models/first_real")