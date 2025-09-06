# Variational Autoencoder for Handwritten Digit Image Generation
This project implements a Variational Autoencoder (VAE), and a Conditional VAE to generate handwritten digit images. The model learns a probabilistic latent representation from the input images, and can generate realistic-looking handwritten digits by sampling from the latent space.

The Conditional VAE extends upon the vanilla VAE by passing in class labels during training and generation. This allows for controlled synthesis of images conditioned on the class label.

## What is the latent space?
The latent space is essentially just a multi-dimensional coordinate system where the representation of input data is held. The model encodes each handwritten digit into the latent space as a Gaussian (normal) distribution, parameterized by mean ($\mu$) and log-variance ($\log \sigma^2$).

This distribution includes the range of possible handwritten variations for a specific digit. 

### The latent vector
A sample taken from the distribution of each digit's representation from the latent space is the latent vector $\vec{z}$.

$\vec{z}$ represents a specific "variation" of a specific handwritten digit.


The ability to sample from a distribution to generate new digit variations makes the model generative.

## How the VAE works
...

## How is Conditional VAE different
...

## The loss function
...





```sh
pyversion 3.12.9

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
matplotlib==3.10.1
```
