import sys, os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from models.vae import VAE
# from models.cvae import CVAE
from trainer import vae_trainer, cvae_trainer
from utils import present_hyperparams

# Check command line argument
if len(sys.argv) != 2 or sys.argv[1] not in ["vae", "conditional_vae"]:
    print("Invalid arguments.\nCorrect usage:")
    print("  python train.py vae                # Train variational autoencoder")
    print("  python train.py conditional_vae    # Train conditional variational autoencoder")
    sys.exit(1)
model_to_train = sys.argv[1]


# The loss function
def the_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# The compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")

# The hyperparameters
hyperparams = {
    'learning_rate': 1e-3,
    'num_epochs': 3,
    'batch_size': 64,
    'device': device
}
present_hyperparams(hyperparams)


# The dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="./", train=True, download=True, transform=transform)

# training and validation splits
dataset_size = len(dataset)
val_size = int(dataset_size * 0.1)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print("\ndataset_size:", dataset_size)
print(f"train_size: {train_size}\tval_size: {val_size}\n")

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}


# The training loop
if model_to_train == "vae":
    model = VAE()
    model, loss_values = vae_trainer.train(model, the_loss_function, hyperparams, dataloaders)
# elif model_to_train == "conditional_vae":
    # model = CVAE()
    # model, loss_values = cvae_trainer.train(model, the_loss_function, hyperparams, dataloaders)


model_name = f"{model_to_train} epoch_{hyperparams['num_epochs']} \
lr_{hyperparams['learning_rate']} bsize_{hyperparams['batch_size']}"

save_path = os.path.join("outputs", model_name)
os.makedirs(save_path, exist_ok=True)


# Save model weights
torch.save(model.state_dict(), os.path.join(save_path, "weights.pth"))

# Plot loss curves
plt.plot(loss_values['train'], label="Training loss")
plt.plot(loss_values['val'], label="Validation loss")
plt.legend()
plt.savefig(os.path.join(save_path, "loss.png"))
