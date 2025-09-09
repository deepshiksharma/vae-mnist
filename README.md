# Variational Autoencoder for Handwritten Digit Image Generation
This project implements a Variational Autoencoder (VAE), and a Conditional VAE to generate handwritten digit images. The model learns a probabilistic latent representation from the input images, and can generate realistic-looking handwritten digits by sampling from the latent space.

The Conditional VAE extends upon the vanilla VAE by passing in class labels during training and generation. This allows for controlled synthesis of images conditioned on the class label.


## How the VAE works
[To be updated]


## How Conditional VAE is different
[To be updated]


## What is the latent space?
The latent space is a multi-dimensional co-ordinate system where the representation of input data is held.

The encoder component of the model encodes each handwritten digit image into the latent space as a normal distribution, parameterized by mean ($\mu$) and standard deviation ($\sigma$).<sup>  _refer note below_</sup> <br>
This distribution includes the range of possible handwritten variations for a specific digit.

<em> Note: In practice, the encoder outputs mean ($\mu$) and specifically, log-variance ($\log \sigma^2$). Predicting $\log \sigma^2$ is more numerically stable and also simplifies the loss calculation. $\sigma$ is derived from $\log \sigma^2$ when needed. For simplicity, I will refer to this quantity as $\sigma$ throughout this readme. </em>

### The latent vector
A sample taken from the distribution of each digit's representation from the latent space is the latent vector $\mathbf{z}$. This vector $\mathbf{z}$ represents a specific "variation" of a specific handwritten digit.

The decoder component of the model takes this latent vector and reconstructs an image, effectively reversing the process of the encoder.
Most reconstructions are synthetic rather than exact copies of the input images. This is because the sampled vectors often lie near (but not exactly on) the original latent embedding for that specific digit.

The ability to sample from a distribution to generate new digit variations makes the VAE model generative.


## The reparameterization trick
Drawing a random sample from a probability distribution is a stochastic operation, and not a smooth mathematical function of $\mu$ and $\sigma$. Directly sampling a latent vector from the encoder's distribution is not differentiable, and would prevent gradients from flowing back into the encoder during training. The operation needs to be differentiable because that’s the only way gradient descent can update parameters $\mu$ and $\sigma$, allowing the encoder to learn useful latent representations.

To solve this, the random sampling of vector $\mathbf{z}$ is re-expressed as a deterministic function of the encoder parameters and an extra random variable term:

$\mathbf{z} = \mu + \sigma \cdot \varepsilon$
- $\mu$ and $\sigma$ are encoder outputs
- $\varepsilon$ is random noise drawn from a standard normal distribution
- $\sigma$ is scaled by $\varepsilon$ to induce variability

This allows the separation of randomness ($\varepsilon$) from trainable parameters ($\mu$ and $\sigma$). $\mathbf{z}$ is now differentiable with respect to $\mu$ and $\sigma$.

The reparameterization trick makes it possible to backpropagate through the random sampling process.


## The loss function
The loss function used to train VAEs is composed of the reconstruction loss (BCE), and the regularization term (KLD):
$\mathcal{L} = \text{BCE} + \text{KLD}$

The reconstruction loss optimizes the decoder to ensure its output resembles real data. Binary cross-entropy is used, which is common when training on normalized grayscale images like MNIST.

$\text{BCE} = - \sum_i \left[ x_i \log(\hat{x}_i) + (1 - x_i)\log(1 - \hat{x}_i) \right]$
- $i$ iterates over all pixels
- $x$ is the original image
- $\hat{x}$ is the reconstructed image

The regularization term is the **Kullback-Leibler divergence** (KLD). KL divergence regularizes the latent space by encouraging latent vectors to be close to a standard normal distribution. This prevents overfitting and makes the latent space continuous.

$\text{KLD} = -\tfrac{1}{2} \sum_j \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$
- $j$ iterates over each latent dimension
- $\log \sigma_j^2$ is the log-variance output from the encoder
- $\mu_j^2$ is the squared mean of latent distribution for dimension $j$
- $\sigma_j^2$ is the variance of latent distribution for dimension $j$



## installation instructions, req.txt., dataset used, training and demo examples
[To be updated]

```sh
pyversion 3.12.9

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
matplotlib==3.10.1
```
