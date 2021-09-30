import tensorflow as tf
import time
from models.cvae import CVAE
from loader import get_training_set
import numpy as np

train_dataset, test_dataset = get_training_set()
batch_size = 32
epochs = 10
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 64
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)
optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, metrics=["mae"])

for test_batch in test_dataset.take(1):
    test_good_sample = test_batch[0:num_examples_to_generate, :, :, :]
    
for epoch in range(1, epochs + 1):
    print("epoch: " + str(epoch))
    start_time = time.time()
    i = 0
    for train_x in train_dataset:
        i += 1
        model.fit(train_x, batch_size=16)
        if i == 2500:
            break
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(model.compute_loss(test_x))
    elbo = -loss.result()
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))

model.save("/home/jerms/daniil/new_vae/saved_models/second_real")
