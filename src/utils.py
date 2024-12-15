import torch
from torchvision import datasets

def load_data():
    mnist_train = datasets.MNIST("./data", train=True, download=True)
    mnist_test = datasets.MNIST("./data", train=False, download=True)

    x_train = mnist_train.data.reshape(-1, 784) / 255
    y_train = mnist_train.targets

    x_test = mnist_test.data.reshape(-1, 784) / 255
    y_test = mnist_test.targets

    return x_train, y_train, x_test, y_test

def get_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> int | float:
    predictions = torch.max(scores, 1).indices
    output = torch.eq(predictions, labels)
    output = torch.where(output == True, 1, 0)
    num_of_correct_predictions = torch.sum(output)
    total_predictions = labels.shape[0]
    accuracy = num_of_correct_predictions / total_predictions
  
    return accuracy
