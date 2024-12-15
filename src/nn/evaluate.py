from src.utils import get_accuracy

def evaluate(model, x_test, y_test):
    """
    Return a number in range [0, 1].
    0 means 0% accuracy while 1 means 100% accuracy.
    """
    scores = model(x_test)
    accuracy = get_accuracy(scores, y_test)
    return accuracy
