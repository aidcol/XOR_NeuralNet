# XOR_NeuralNet
This repository contains a simple feedfoward neural network (random weight initialization and no regularization) that classifies XOR. The Jupyter notebook contains the XOR_Net class and demonstrates its use with a successful choice of hyperparamters in subsequent cells. There is also a Python file containing just the XOR_Net class.

## Network Architecture
The architecture of the network is as follows:
- Input layer: 2 input neurons
- Hidden layer: 2 hidden neurons
- Output layer: 1 output neuron

The network implements cross-entropy cost with sigmoid activation. Activation of the output neuron determines the binary classification (<0.5 is 0, >=0.5 is 1). Training of the network is done by gradient descent using backpropagation, which is implemented from scratch using NumPy.

## Resources
The code for this network is based on the implementation in the following [online text](http://neuralnetworksanddeeplearning.com/).
Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press 2015.

The open-source code in the online text can also be found in Michael Nielsen's own [online repository](https://github.com/mnielsen/neural-networks-and-deep-learning).