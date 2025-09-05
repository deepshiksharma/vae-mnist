import sys
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
    sys.exit("Incorrect usage.")
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
    'num_epochs': 100,
    'batch_size': 128,
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
print("dataset_size:", dataset_size)
print(f"train_size: {train_size}\tval_size: {val_size}\n")

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}


# The training loop
if model_to_train == "vae":
    model = VAE()
    trained_model, loss_values = vae_trainer.train(model, the_loss_function, hyperparams, dataloaders)
elif model_to_train == "conditional_vae":
    model = CVAE()
    trained_model, loss_values = cvae_trainer.train(model, the_loss_function, hyperparams, dataloaders)

# Save model weights
torch.save(trained_model.state_dict(), f"{model_to_train}.pth")

# Plot loss curves
plt.plot(loss_values['train'], label="Training loss")
plt.plot(loss_values['val'], label="Validation loss")
plt.legend()
plt.savefig(f"{model_to_train} loss curves.png")
