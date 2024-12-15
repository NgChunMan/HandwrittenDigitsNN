import torch
import torch.nn as nn
from src.DigitNet import DigitNet

def train_nn_model(x_train: torch.Tensor, y_train: torch.Tensor, epochs=20):
    model = DigitNet(784, 10)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_train)

        # Compute loss
        loss = loss_fn(y_pred, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    return model
