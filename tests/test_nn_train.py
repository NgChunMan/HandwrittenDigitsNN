import torch
from src.nn.model import DigitNet
from src.nn.train import train_nn_model

def test_nn_train():
    x_train_new = torch.rand(5, 784, requires_grad=True)
    y_train_new = ones = torch.ones(5, dtype=torch.uint8)
    assert type(train_nn_model(x_train_new, y_train_new)) == DigitNet
