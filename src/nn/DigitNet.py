import torch.nn as nn

class DigitNet(nn.Module):
    def __init__(self, input_dimensions: int, num_classes: int):
        super().__init__()
        self.l1 = nn.Linear(input_dimensions, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.softmax(self.l3(x))
        return x
