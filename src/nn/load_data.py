from torchvision import datasets

def load_data_nn():
    # This downloads the MNIST datasets ~63MB
    mnist_train = datasets.MNIST("./data", train=True, download=True)
    mnist_test  = datasets.MNIST("./data", train=False, download=True)
    
    x_train = mnist_train.data.reshape(-1, 784) / 255
    y_train = mnist_train.targets
    
    x_test = mnist_test.data.reshape(-1, 784) / 255
    y_test = mnist_test.targets

    return x_train, y_train, x_test, y_test
