import sys
import torch
import matplotlib.pyplot as plt
from models.vae import VAE
from models.cvae import CVAE


# Check command line argument for which model to sample from
if len(sys.argv) < 2 or sys.argv[1] not in ["vae", "conditional_vae"]:
    print("Invalid arguments.\nCorrect usage:")
    print("  python sampler.py vae                  # Sample random digits from variational autoencoder")
    print("  python sampler.py conditional_vae 7    # Sample digit 7 from conditional variational autoencoder")
    sys.exit(1)
model_to_sample = sys.argv[1]


# The conditional VAE needs a digit to condition on, the vanilla VAE does not
digit = None
if model_to_sample == "conditional_vae":
    if len(sys.argv) != 3 or sys.argv[2] not in [str(d) for d in range(10)]:
        sys.exit("Specify which digit to generate, from 0 to 9:\n  python sampler.py conditional_vae 7")
    digit = int(sys.argv[2])
elif len(sys.argv) != 2:
    sys.exit("The vanilla VAE samples digits at random, it takes no digit argument.")


# Load trained model weights
VAE_WEIGHTS = "outputs/vae epoch_10 lr_0.001 bsize_64/weights.pth"
CVAE_WEIGHTS = "outputs/conditional_vae epoch_10 lr_0.001 bsize_64/weights.pth"

LATENT_DIM = 20     # the default, as defined in models/vae.py and models/cvae.py
NUM_CLASSES = 10    # 10 digit classes of the MNIST dataset
NUM_SAMPLES = 9     # displayed as a 3x3 grid


# The compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# Load the trained model
if model_to_sample == "vae":
    model = VAE().to(device)
    model.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
else:
    model = CVAE(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(CVAE_WEIGHTS, map_location=device))
model.eval()


# Sample latent vectors from the prior, and decode them into images
with torch.no_grad():
    z = torch.randn(NUM_SAMPLES, LATENT_DIM, device=device)
    if model_to_sample == "vae":
        images = model.decoder(z).cpu()
        title = "VAE generated samples"
    else:
        labels = torch.full((NUM_SAMPLES,), digit, dtype=torch.long, device=device)
        images = model.decoder(z, labels).cpu()
        title = "Conditional VAE generated samples"


# Display the generated images
fig, axes = plt.subplots(3, 3, figsize=(4.5, 4.5))
fig.suptitle(title)
for image, ax in zip(images, axes.flat):
    ax.imshow(image.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
fig.tight_layout()
plt.show()
