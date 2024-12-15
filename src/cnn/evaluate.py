import torch
from src.utils import get_accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_cnn(model, test_loader, device):
    """
    Evaluate model accuracy on the test dataset.
    """
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            pred_model = model(x)
            accuracy = get_accuracy(pred_model, y)
            print(f"drop-out (0.5) accuracy: {accuracy}")
            return accuracy
