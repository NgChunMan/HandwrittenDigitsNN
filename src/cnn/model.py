import torch
import torch.nn as nn

class DropoutCNN(nn.Module):
    """
    CNN that uses Conv2d, MaxPool2d, and Dropout layers.
    """
    def __init__(self, classes: int, drop_prob: float = 0.5):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for MNIST
        drop_prob: probability of dropping a node in the neural network
        """
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.l1 = nn.Linear(1600, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, classes)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(p=drop_prob)
        self.lrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        
        x = x.view(-1, 64*5*5) # Flattening

        x = self.l1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.lrelu(x)
        x = self.l3(x)
        return x
