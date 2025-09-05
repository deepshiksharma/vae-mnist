# from ..models import 

def train(hyperparams, dataloaders):
    # hyperparams
    lr = hyperparams['learning_rate']
    num_epochs = hyperparams['num_epochs']
    device = hyperparams['device']
    
    # dataloaders
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    ...
    
    loss_values = {'train':, 'val':}
    
    return model, loss_values
