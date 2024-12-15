# MNIST Handwritten Digits Recognition

This project implements classification of handwritten digits (0-9) from the MNIST dataset using two distinct approaches:

1. Fully-connected Linear Neural Network (FCNN)
2. Convolutional Neural Network (CNN) with Dropout

The primary goal is to compare the performance of these models in terms of their ability to generalize and accurately classify unseen data. The MNIST dataset, a benchmark dataset for handwritten digit recognition, consists of grayscale images of digits (0-9) with a resolution of 28x28 pixels.

## Project Overview

The project implements and evaluates two types of neural networks:

Fully-Connected Neural Network (FCNN): This model uses fully connected linear layers to classify the digits.

Convolutional Neural Network (CNN) with Dropout: This model leverages convolutional layers to capture spatial hierarchies in the images and applies dropout to prevent overfitting.

By comparing these models, we demonstrate the advantages of CNNs, particularly with dropout, in achieving better generalization and accuracy on image data.

## Methodology
1. Model Architectures:
- FCNN: A simple neural network with fully connected layers.
- CNN with Dropout: A convolutional neural network with added dropout layers for regularization.

2. Training:
- Models are trained using the Adam optimizer and CrossEntropy loss function.
- CNN is trained for 10 epochs with a batch size of 256.

3. Evaluation:
- Accuracy is computed by comparing predictions with ground-truth labels on the test set.

## Setup Instructions
1. Clone the Repository:
```
git clone https://github.com/NgChunMan/HandwrittenDigitsNN.git
cd HandwrittenDigitsNN
```

2. Install required Dependencies:
```
pip install -r requirements.txt
```

3. Download the following dataset and place them in the `data/MNIST/raw` directory:
- [train-images-idx3-ubyte](https://drive.google.com/file/d/1SX7puzoeKgPRfKnys7Kmh3OeM1rWOsmw/view?usp=share_link)

4. Run the main script to train the model and classify transactions:
```
python main.py
```

## Testing
Unit tests are provided to validate the implementation. Run the tests using pytest:
```
python -m tests.test_nn_evaluate
python -m tests.test_nn_model
python -m tests.test_nn_train

```

## Results

### Fully-Connected Neural Network (FCNN) Model:
- Accuracy: 76%
- The FCNN model struggles to generalize, achieving an accuracy of only 76% on the test set. This indicates its limited capacity to capture spatial patterns in the images.

### Convolutional Neural Network (CNN) with Dropout:
- Epoch-wise Losses:
  ```
  Epoch: 0, Loss: 0.45955332993192877
  Epoch: 1, Loss: 0.13243763763853844
  Epoch: 2, Loss: 0.10421276759514783
  Epoch: 3, Loss: 0.08661896473866829
  Epoch: 4, Loss: 0.0778915832889207
  Epoch: 5, Loss: 0.07346109235064781
  Epoch: 6, Loss: 0.06521662709877846
  Epoch: 7, Loss: 0.06408926895720528
  Epoch: 8, Loss: 0.058425070927973756
  Epoch: 9, Loss: 0.05647834554076829
  ```
- Accuracy (with drooout probability=0.5): 99.1%
- The CNN model with dropout achieves a significantly higher accuracy of 99.1%. This demonstrates its superior ability to generalize by effectively leveraging spatial features and regularization.
