import sys
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import trainer
from utils import present_hyperparams

# Check command line argument
model_to_train = sys.argv[1]
if model_to_train not in ["vae", "conditional_vae"]:
    sys.exit("Incorrect usage.")

# The compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")

# The hyperparameters
hyperparams = {
    'learning_rate': 1e-3,
    'num_epochs': 80,
    'batch_size': 64,
    'device': device
}
present_hyperparams()


# The dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="./", train=True, download=True, transform=transform)

# training and validation splits
dataset_size = len(dataset)
val_size = int(dataset_size * 0.1)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print("dataset_size:", dataset_size)
print(f"train_size: {train_size}\tval_size: {val_size}")

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}


# The training loop
if model_to_train == "vae":
    trained_model, loss_values = trainer.vae.train(hyperparams, dataloaders)

elif model_to_train == "conditional_vae":
    trained_model, loss_values = trainer.c_vae.train(hyperparams, dataloaders)


# Save model weights
torch.save(trained_model.state_dict(), f"{model_to_train}.pth")

# Plot loss curves
plt.plot(loss_values['train'], label="Training loss")
plt.plot(loss_values['val'], label="Validation loss")
plt.legend()
plt.savefig(f"{model_to_train} loss curves.png")
