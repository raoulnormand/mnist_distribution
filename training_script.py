import tensorflow as tf
from tensorflow import keras
from keras import layers
from train import create_dataset, train_model

# Constants

input_size = (28, 28)

x_train, x_val = create_dataset()

model = tf.keras.models.Sequential([
    layers.Conv2D(64, 7, activation='swish', padding='same', input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D(2),
    layers.Conv2D(128, 3, activation='swish', padding='same', name='conv2'),
    #layers.Conv2D(128, 3, activation='swish', padding='same', name='conv3'),
    layers.MaxPooling2D(2),
    layers.Conv2D(256, 3, activation='swish', padding='same', name='conv4'),
    #layers.Conv2D(256, 3, activation='swish', padding='same', name='conv5'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='swish', name='dense1'),
    layers.Dense(1, name='dense2', use_bias=False)
])

optimizer = keras.optimizers.SGD(learning_rate=1e-3, clipvalue=0.1)

train_model(model, x_train, x_val, batch_size=128, n_epochs=4, optimizer=optimizer, reg_param=1e-5, input_size=input_size,
n_langevin_steps=100, step_size=0.01, n_samples=10, CD=False)

# Get samples

import matplotlib.pyplot as plt
from train import langevin_samples

samples = langevin_samples(model, 100, 0.01, 1, input_size)
plt.imshow(samples[0])
