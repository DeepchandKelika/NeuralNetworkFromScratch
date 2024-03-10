# Neural Network Implementation from Scratch

## Overview
This repository contains a detailed implementation of a neural network built from scratch using Python. The purpose is to demonstrate the underlying mechanics of neural networks without relying on high-level machine learning libraries.
Fundamentals of Neural Networks

Before diving into implementation, let's understand some basic concepts:

- Neuron: The fundamental unit of a neural network, which receives input, processes it, and passes an output to the next layer.
- Weights and Biases: Parameters of the model that are adjusted during training to minimize the loss function.
- Activation Function: A mathematical operation applied to a neuron's output to introduce non-linearity into the network, enabling it to learn complex patterns.
- Loss Function: A method to measure how well the neural network performs, guiding the update of weights through backpropagation.
- Backpropagation: The core algorithm of neural network training, responsible for adjusting the weights and biases to minimize the loss.
- Learning Rate: A parameter that determines the step size at each iteration while moving toward a minimum of the loss function

- <b>Common Activation Functions</b>

    - Sigmoid Function: Traditionally used, especially for the output layer in binary classification problems. It maps any input into a value between 0 and 1. However, it's not typically used for hidden layers anymore due to its drawbacks like vanishing gradients.<br>
    `σ(z) = 1 / (1 + e^(-z))`

    - ReLU (Rectified Linear Unit): Currently, one of the most popular activation functions for hidden layers. It's computationally efficient and allows models to converge faster by overcoming the vanishing gradient problem, to some extent.<br>
   `ReLU(z) = max(0, z)`

    - Tanh (Hyperbolic Tangent): Similar to the sigmoid but maps the input to values between -1 and 1, which can be more beneficial in certain scenarios compared to the sigmoid function, as it centers the data, improving the efficiency of the learning process.<br>
   `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`

    - Softmax: Used in the output layer of a multi-class classification problem. It provides the probabilities for each class, with the sum of all probabilities equaling one.<br>
   `Softmax(z_i) = e^(z_i) / Σ(e^(z_j))` (Summation for j over all classes)

## Features
- Layer-wise architecture
- Various activation functions (Sigmoid, ReLU, Tanh, Softmax )
- Different loss functions (Mean Squared Error (MSE), Binary Cross-Entropy Loss, Categorical Cross-Entropy Loss)
- Implementation of forward and backward propagation

## Prerequisites
- Ensure you have Python installed on your system. 
- Install numpy library: pip install python


```bash
git clone https://github.com/yourusername/neural-network-from-scratch.git
cd neural-network-from-scratch
