"""
Functions to train the Energy-Based Model
and obtain samples
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Create the dataset

def create_dataset(numbers=None):
    """
    Creates the dataset made up of images of the selected numbers,
    and creates batches of the corresponding size.
    _ numbers: list / array / ... of numbers to pick
    _ batch_size: the batch size
    There is no need for a test set, so we use the second part of the
    data as validation set to monitor overfitting.
    """
    (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
    if numbers is not None:
        x_train = x_train[np.isin(y_train, numbers)]
        x_val = x_val[np.isin(y_val, numbers)]
    x_train = x_train/255
    x_val = x_val/255
    # Add extra dimension in order to use CNN with keras
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    return x_train, x_val

# Langevin sampling with model

def langevin_samples(model, n_langevin_steps, step_size, n_samples, input_size, initial_state=None):
    """
    Provides a sample from the Gibbs distribution with energy given by model,
    using a discretized Langevin diffusion.
    _ model: a (multilayer) (C)NN
    _ n_langevin steps: number of steps to take
    _ step_size: discretization step
    _ n_samples: number of independent samples to obtain
    _ initial_state: the starting state of the diffusion.
    It is uniformly random in [0, 1] by default. Otherwise, it can be given as a
        numpy array.
    """
    # Initial state
    if initial_state is None:
        samples = tf.Variable(np.random.rand(n_samples, *input_size, 1), dtype='float32')
    else:
        samples = tf.Variable(initial_state, dtype='float32')
    # Run the diffusion
    for _ in range(n_langevin_steps):
        # Compute the gradient of the energy
        with tf.GradientTape() as tape:
            energy_samples = model(samples)
        grad_samples = tape.gradient(energy_samples, samples)
        # Update according to the discretized diffusion
        samples.assign_sub(step_size*grad_samples)
        samples.assign_add(np.sqrt(2*step_size)*np.random.normal(size=(n_samples, *input_size, 1)))
        # Clip at 0 and 1 to keep values in the correct range.
        samples.assign(tf.clip_by_value(samples, clip_value_min=0, clip_value_max=1))
    return samples

# Create custom training loop

def train_model(model, x_train, input_size, batch_size, n_epochs,
        optimizer, n_langevin_steps, step_size, n_samples, reg_param=0,
        use_cd=False, print_grad=False):
    """
    Trains the model with the given parameters.
    _x_train: the training data
    _batch_size: the batch size, with remainder dropped
    _optimizer: a keras optimizer
    _reg_param: the regularization parameter. For ease of use,
    it is the same for all layers. It is multiplied by the
    learning rate of the optimizer.
    _n_langevin_steps, step_size, n_samples: as for the
    langevin_samples function
    _use_cd: whether to use Contrastive Divergence. In this case,
    the Langevin samples are initialized using the data of the batch,
    and n_samples is set to batch_size.
    _print_grad: whether to print the l2 norm of the gradient.
    """
    batches = tf.data.Dataset.from_tensor_slices(x_train)
    batches = batches.shuffle(buffer_size=x_train.shape[0])
    batches = batches.batch(batch_size=batch_size, drop_remainder=True)
    # Iterate over epochs
    for epoch in range(n_epochs):
        print(f'\nStart of epoch {epoch+1}/{n_epochs}')
        if use_cd:
            n_samples = batch_size
        # Iterate over batches
        for step, batch in enumerate(batches):

        # Compute the gradient of the energy of a batch
            with tf.GradientTape() as tape:
                energy_batch = model(batch)
            grad_batch = tape.gradient(energy_batch, model.trainable_weights)
            grad_batch = [g / batch_size for g in grad_batch]

            # Get samples according to the model distribution
            if use_cd:
                samples = langevin_samples(model, n_langevin_steps, step_size,
                n_samples=batch_size, input_size=input_size, initial_state=batch.numpy())
            else:
                samples = langevin_samples(model, n_langevin_steps, 
                                step_size, n_samples, input_size)
            
            # Compute the gradient of the energy of the samples
            with tf.GradientTape() as tape:
                energy_samples = model(samples)
            grad_samples = tape.gradient(energy_samples, model.trainable_weights)
            grad_samples = [g / n_samples for g in grad_samples]
            
            # Make a gradient step
            gradients = [gb - gs for gb, gs in zip(grad_batch, grad_samples)]
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            # Deal with l2 regularization
            if reg_param != 0:
                for weights in model.trainable_weights:
                    weights.assign_sub(optimizer.learning_rate * reg_param  * weights)

            if step % 5 == 0:
                print(f"\nStep {step+1}/{x_train.shape[0]//batch_size}")
                if print_grad:
                    print("l2 norm gradient", [tf.norm(g).numpy() for g in gradients])
