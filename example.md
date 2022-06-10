# Example of sample obtained from a trained model

- First import the necessary modules.

```
import matplotlib.pyplot as plt
from tensorflow import keras
from train import langevin_samples, sequential_langevin_samples
```

- A model trained on the number 3 is available in the trained_model_only_3. Load it with

```
model = keras.models.load_model('trained_model_only_3')
```

- Obtain samples and display them with the following command. The number of samples, parameters of the diffusion, and initial state can be toyed with.

```
samples = langevin_samples(model, 100, 0.01, 5, input_size)
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for sample, ax in zip(samples, axes.flat):
    ax.imshow(sample, cmap='gray')
```

![5 samples of the number 3](/images/samples_3.png)


- Alternatively, you can show the evolution of the Langevin samples. Here, 200 steps are taken, and the state is shown every 20 images.

```
samples = sequential_langevin_samples(model, 10, 10, 0.01, input_size)
fig, axes = plt.subplots(2, 5, figsize=(15, 5))
for sample, ax in zip(samples, axes.flat):
    ax.imshow(sample, cmap='gray')
```

![Seqwuential samples](/images/sequential_samples.png)