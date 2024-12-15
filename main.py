import torch
from src.cnn.evaluate import evaluate_cnn
from src.cnn.load_data import load_data_cnn
from src.cnn.model import DropoutCNN
from src.cnn.train import train_cnn_model

from src.nn.evaluate import evaluate_nn
from src.nn.load_data import load_data_nn
from src.nn.train import train_nn_model

# Set up device (cuda or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data for NN
print("Loading data for Neural Network...")
x_train, y_train, x_test, y_test = load_data_nn()

# Load data for CNN
print("Loading data for Convolutional Neural Network...")
train_loader, test_loader = load_data_cnn()

# Train and evaluate Neural Network
try:
    print("======Training Neural Network Model======")
    digit_model = train_nn_model(x_train, y_train)
    
    print("======Evaluating Neural Network Model======")
    accuracy_digit_model = evaluate_nn(digit_model, x_test, y_test)
    print(f"Neural Network Model Accuracy: {accuracy_digit_model:.2f}")
except Exception as e:
    print(f"Error during Neural Network processing: {e}")

# Train and evaluate CNN with Dropout
try:
    print("======Training Dropout Model======")
    do_model = DropoutCNN(10)
    do_model, losses = train_cnn_model(train_loader, do_model, device)
    
    # Evaluate model
    print("======Evaluating Dropout Model======")
    accuracy_do_model = evaluate_cnn(do_model, test_loader, device)
    print(f"Dropout Model Accuracy: {accuracy_do_model:.2f}")
except Exception as e:
    print(f"Error during CNN processing: {e}")
