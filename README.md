 # Neural Network with Four Hidden Layers in C++

This repository contains a C++ implementation of a deep neural network with four hidden layers. The network is designed for multi-class classification tasks and is trained using backpropagation with stochastic gradient descent.

## Key Features:

  Four Hidden Layers: The network architecture includes four hidden layers with configurable neuron counts.
  
  Softmax Activation: Softmax activation function is used in the output layer for multi-class classification.
  
  Backpropagation: Implements the backpropagation algorithm for efficient weight updates.
  
  Random Weight Initialization: Weights are initialized using a normal distribution to break symmetry and speed up convergence.
  
  Training and Prediction: Provides methods for training the network on input data with corresponding labels, and making predictions on new data.

## Usage:

1. Include the Header File: Include `NeuralNetwork.h` in your C++ project.

2. Create a Network Instance: Initialize a `NeuralNetwork` object with the desired input size, hidden layer sizes, output size, and learning rate.

3. Train the Network: Use the `train` method to train the network on your dataset.

4. Make Predictions: Use the `predict` method to obtain predictions for new input data.
