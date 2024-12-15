%%time
import torch
import torch.nn as nn

def train_model(loader: torch.utils.data.DataLoader, model: nn.Module, device: torch.device):
    optimiser = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    epoch_losses = []
    model = model.to(device)
    for i in range(10):
        epoch_loss = 0
        model.train()
        for idx, data in enumerate(loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimiser.step()
            epoch_loss = epoch_loss + loss.item()

        epoch_loss = epoch_loss / len(loader)
        epoch_losses.append(epoch_loss)
        print ("Epoch: {}, Loss: {}".format(i, epoch_loss))

    return model, epoch_losses
