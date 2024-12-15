import torch
from torchvision import datasets

def get_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> int | float:
    predictions = torch.max(scores, 1).indices
    output = torch.eq(predictions, labels)
    output = torch.where(output == True, 1, 0)
    num_of_correct_predictions = torch.sum(output)
    total_predictions = labels.shape[0]
    accuracy = num_of_correct_predictions / total_predictions
  
    return accuracy
