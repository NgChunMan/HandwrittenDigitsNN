from src.nn.model import DigitNet

def test_nn_model():
    model = DigitNet(784, 10)
    assert [layer.detach().numpy().shape for name, layer in model.named_parameters()] \
        == [(512, 784), (512,), (128, 512), (128,), (10, 128), (10,)]
