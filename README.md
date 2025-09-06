# Variational Autoencoder for Handwritten Digit Image Generation
This project implements a Variational Autoencoder (VAE), and a Conditional VAE to generate handwritten digit images. The model learns a probabilistic latent representation of images, and can generate realistic-looking images by sampling from the latent space.

The Conditional VAE extends upon the vanilla VAE by passing in class labels during training and generation. This allows for controlled synthesis of images conditioned on the class label.

## What is the latent space?
The latent space is essentially a multi-dimentional coordinate system where the representation of input data is held.

### The latent vector
The model encodes each handwritten digit into the latent space as a Gaussian (normal) distribution. This distribution includes the range of possible handwritten variations for a specific digit.

## How the VAE works
...

### How is Conditional VAE different
...

### The loss function
...





```sh
pyversion 3.12.9

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
matplotlib==3.10.1
```
