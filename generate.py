"""
Generates the sample grids used as figures in README.md.
Run from the repository root, after both models have been trained:

    python make_assets.py
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.vae import VAE
from models.cvae import CVAE


# Trained model weights
VAE_WEIGHTS = "outputs/vae epoch_10 lr_0.001 bsize_64/weights.pth"
CVAE_WEIGHTS = "outputs/conditional_vae epoch_10 lr_0.001 bsize_64/weights.pth"

LATENT_DIM = 20         # as defined in models/vae.py and models/cvae.py
NUM_CLASSES = 10        # 10 digit classes of the MNIST dataset
VAE_SAMPLES = 9         # displayed as a 3x3 grid
CVAE_SAMPLES = 10       # per digit, displayed as a 10x10 grid

ASSETS_DIR = "assets"
DPI = 300
ROW_LABELS = True       # label each row of the conditional VAE grid with its digit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")


def save_grid(images, rows, cols, filename, row_labels=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for image, ax in zip(images, axes.flat):
        ax.imshow(image.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    if row_labels is not None:
        for row, label in enumerate(row_labels):
            axes[row][0].set_ylabel(label, rotation=0, labelpad=12, va="center", fontsize=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    path = os.path.join(ASSETS_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"saved {path}")


os.makedirs(ASSETS_DIR, exist_ok=True)


# The vanilla VAE: latent vectors drawn at random, no control over the class
vae = VAE(latent_dim=LATENT_DIM).to(device)
vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
vae.eval()

with torch.no_grad():
    z = torch.randn(VAE_SAMPLES, LATENT_DIM, device=device)
    images = vae.decoder(z).cpu()

save_grid(images, 3, 3, "vae_samples.png")
