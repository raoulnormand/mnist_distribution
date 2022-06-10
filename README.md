# An Energy-Based Model for the MNIST digit distribution

The goal of this project is two-fold.

1. Train an Energy-Based Model on the MNIST digits distribution.
2. Sample from this distribution to obtain fake handwritten digits.

The distribution on 28 x 28 images is modeled by an energy-based model (EBM), see Section 16.2.4 in [[1]](#1). The energy itself described by a neural network made of several convolutional layers followed by a few dense layers, with a scalar output. We aim at minimizing the negative log-likelihood. To this end, we need to compute its gradient. It consists of two parts, as described in Ch. 18 of [[1]](#1).

1. A **positive** part, which is the gradient of the energy of the data (or a batch), and can be obtained by autodifferentiation.
2. A **negative** part, which the gradient of the energy of a sample obtained from the model distribution.

Therefore, the main difficulty is the negative part, for which we need to obtain samples from the model distribution. This can be done using a discretized overdampled Langevin dynamics (sans Metropolis-Hastings step, aka MALA), as described in [[2]](#2).

## Technologies

Project created with:
- Python 3.8
- Numpy 1.20.1
- Tensorflow 2.8.0
- Keras 2.8.0

## Requirements

To install the requirements for the project, run
```
pip install -r requirements.txt
```

## Functions in train.py

The train.py module contains the main functions to train the model and get samples.

- First, create the dataset by running

```
x_train, x_val = create_dataset(numbers)
```

where `numbers` is an optional argument describing which parts of the dataset to load. For instance, `numbers=[5, 7]` loads only images of 5 and 7. This defaults to None to load the whole dataset. Smaller datasets require less training and hyperparameter tweaking. Note than an extra dimension is added in order to use CNN in keras. As no validation is used, it is possible to concatenate both datasets to obtain a larger training set.

- The functions require an argument input_size, which is the size in pixels of the images, so `input_size=(28, 28)` for the mnist digits.

- Assuming that a model is given (describing the energy), samples from the distribution given by this energy can be obtained with

```
langevin_samples(model, n_langevin_steps, step_size, n_samples, input_size, initial_state=None)
```

The second to fourth arguments control the discretized Langevin dynamics: the number of steps, the discretization parameter, and the number of samples obtained. `initial_state` can be any numpy array of the correct size, or defaults to a uniform distribution in $[0,1]$.

- Create a keras model (see an example in training_script.py), and train it with

```
train_model(model, x_train, batch_size, n_epochs, optimizer, reg_param, input_size,
        n_langevin_steps, step_size, n_samples, use_cd, print_grad)
```

The arguments `n_langevin_steps`, `step_size`, `n_samples` control the discretized Langevin dynamics used to estimate the negative part of the gradient.

To use Contrastive Divergence, set `use_cd = True`. In this case, the Langevin samples are initialized using the data of the batch and n_samples is set to batch_size.

Set `print_grad=True` to print the $$l^2$$-norm of the gradient. If the model distribution is exactly that of the data, there will be no gradient in expectation. Experimentally, probably due to the architecture of the model, the norm first increases quite a lot, and when ti statibilizes, some fairly good samples can be obtained.

- Once the model is trained, enjoy your new digits with
````
samples = langevin_samples(model, n_langevin_steps, step_size, n_samples, input_size, initial_state=None)
plt.imshow(samples[i], cmap='gray') #Where i is any number in range(n_samples).
````

- It is also possible to obtain samples at different steps of the Langevin dynamics in order to see the evolution. To this end, use
```
samples = sequential_langevin_samples(model, 20, 10, 0.01, input_size)
```

## References
<a id="1">[1]</a>
http://www.deeplearningbook.org

<a id="2">[2]</a>
https://arxiv.org/abs/1903.08689