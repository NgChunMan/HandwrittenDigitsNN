import torch
from src.utils import get_accuracy
from numpy import isclose

def test_nn_evaluate():
    scores = torch.tensor([[0.4118, 0.6938, 0.9693, 0.6178, 0.3304, 0.5479, 0.4440, 0.7041, 0.5573,
             0.6959],
            [0.9849, 0.2924, 0.4823, 0.6150, 0.4967, 0.4521, 0.0575, 0.0687, 0.0501,
             0.0108],
            [0.0343, 0.1212, 0.0490, 0.0310, 0.7192, 0.8067, 0.8379, 0.7694, 0.6694,
             0.7203],
            [0.2235, 0.9502, 0.4655, 0.9314, 0.6533, 0.8914, 0.8988, 0.3955, 0.3546,
             0.5752],
            [0,0,0,0,0,0,0,0,0,1]])
    y_true = torch.tensor([5, 3, 6, 4, 9])
    acc_true = 0.4
    assert isclose(get_accuracy(scores, y_true),acc_true) , "Mismatch detected"
    print("passed")
