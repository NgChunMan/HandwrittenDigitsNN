# MNIST Handwritten Digits Recognition

This project implements classification of handwritten digits (0-9) from the MNIST dataset using two distinct approaches:

1. Fully-connected Linear Neural Network (NN)
2. Convolutional Neural Network (CNN) with Dropout

The aim is to compare the performance of these models and demonstrate the use of dropout to enhance generalization in CNNs.

##Features
- Dataset:
  - The project uses the classic MNIST Handwritten Digits dataset.
- Neural Network (NN):
  - A fully-connected neural network is implemented.
  - Trains on raw pixel data from MNIST.
- Convolutional Neural Network (CNN):
  - Uses convolutional layers with dropout for better generalization.
  - Designed to handle image data more efficiently.
- Comparison:
  - Model accuracy and training performance are compared between the NN and the CNN.
- Device Support:
  - Leverages GPU acceleration with PyTorch when available.

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

3. Download the following dataset and place them in the `data/` directory:
- [credit_card.csv](https://drive.google.com/file/d/1DXAtZnr-mrHccmMX6k1NRssRz2T889G3/view?usp=drivesdk)

4. Run the main script to train the model and classify transactions:
```
python main.py
```

## Testing
Unit tests are provided to validate the implementation. Run the tests using pytest:
```
pytest tests/
```

## Results










