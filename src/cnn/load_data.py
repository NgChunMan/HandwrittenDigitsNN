import torch
from torchvision import datasets, transforms

def load_data():
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    mnist_train = datasets.MNIST("./data", train=True, download=False, transform=T)
    mnist_test = datasets.MNIST("./data", train=False, download=False, transform=T)
    
    train_loader = torch.utils.data.DataLoader(mnist_train, shuffle=True, batch_size=256)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=10000)

    return train_loader, test_loader
