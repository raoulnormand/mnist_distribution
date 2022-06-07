# An Energy-Based Model for the MNIST digit distribution

The goal of this project is two-fold.

1. Train an Energy-Based Model on the MNIST digits distribution.
2. Sample from this distribution to obtain fake handwritten digits.

The distribution on 28 x 28 images is modeled by an energy-based model (EBM), see Section 16.2.4 in [1][1]. The energy itself described by a neural network made of several convolutional layers followed by dense layers, and we aim at minimizing the negative log-likelihood. This can be achieved by computing its gradient, that consists of two parts, see Ch. 18 in [1][1].

1. The gradient of the energy, obtained by autodifferentiation.
2. The gradient of the energy of a sample obtained from the distribution.

Samples are obtained using overdampled Langevin dynamics (sans Metropolis-Hastings step, aka MALA), as described in [2][2].

## Technologies

Project created with:
- Python 3.8
- Numpy 1.20.1
- Tensorflow 2.8.0
- Keras 2.8.0
- Matplotlib 3.3.4

## Requirements

To install the requirements for the project, run

$ pip install -r requirements.txt

## Setup

- Create batches of the dataset by running

$ create_dataset(batch_size, num)

num is an optional argument describing which parts of the dataset to load. For instance, num=3 loads only images of 3, and num=[5, 7] only images of 5 and 7. Defaults to None to load the whole dataset. Smaller datasets require less training and hyperparameter tweaking.

- Create a keras model.

- Train it with

$ train(model, learning_rate, n_epochs, n_langevin_steps, eps, n_samples)

The 3 last arguments control the discretized Langevin dynamics: the number of steps, the discretization parameter, and the number of samples used to estimate the gradient.

- Once the model is trained, you can obtain new samples with

$ create_samples(model, n_langevin_steps, eps, n_samples, show_sanples)

The number of samples can be an int or a couple of int. If True (default), the last argument displays the samples obtained, in the shape given by n_samples (a row if n_samples is an int).

- It is also possible to obtain samples at different steps of the Langevin dynamics in order to see the evolution. To this end, use

$create_sequential_samples(models, eps, n_samples, n_steps, show_samples)

n_steps controls the number of steps between each sample.

[1]: http://www.deeplearningbook.org
[2]: https://arxiv.org/abs/1903.08689