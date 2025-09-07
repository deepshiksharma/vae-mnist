# Variational Autoencoder for Handwritten Digit Image Generation
This project implements a Variational Autoencoder (VAE), and a Conditional VAE to generate handwritten digit images. The model learns a probabilistic latent representation from the input images, and can generate realistic-looking handwritten digits by sampling from the latent space.

The Conditional VAE extends upon the vanilla VAE by passing in class labels during training and generation. This allows for controlled synthesis of images conditioned on the class label.


## How the VAE works
[work in progress]


## How Conditional VAE is different
[work in progress]


## What is the latent space?
The latent space is a multi-dimensional coordinate system where the representation of input data is held.

The encoder component of the model encodes each handwritten digit image into the latent space as a normal distribution, parameterized by mean ($\mu$) and variance ($\sigma^2$). This distribution includes the range of possible handwritten variations for a specific digit. 

### The latent vector
A sample taken from the distribution of each digit's representation from the latent space is the latent vector $\mathbf{z}$.
The vector $\mathbf{z}$ represents a specific "variation" of a specific handwritten digit.

The decoder component of the model takes this latent vector and reconstructs an image, effectively reversing the process of the encoder.
Most reconstructions are synthetic rather than exact copies of the input images. This is because the sampled vectors often lie near (but not exactly on) the original latent embedding for that specific digit.

The ability to sample from a distribution to generate new digit variations makes the VAE model generative.


## The loss function
[work in progress]



---
## install instructions, req.txt. training and demo examples
[work in progress]

```sh
pyversion 3.12.9

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
matplotlib==3.10.1
```

