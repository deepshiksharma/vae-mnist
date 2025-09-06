import torch
from torch import optim

def train(model, the_loss_function, hyperparams, dataloaders):
    lr = hyperparams['learning_rate']
    num_epochs = hyperparams['num_epochs']
    device = hyperparams['device']
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = list(), list()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = the_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        avg_train_loss = train_loss/len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss = the_loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()
            avg_val_loss = val_loss/len(val_loader.dataset)
            val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    loss_values = {'train':train_losses, 'val':val_losses}
    return model, loss_values
