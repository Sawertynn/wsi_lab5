import numpy as np


def laplace(x, mu=0, b=1):
    return (1 / (2 * b)) * np.exp(-np.abs(x - mu) / b)


class Perceptron_2_layers:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        # TO DO: Initialize weights and biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, x):
        # TO DO: Implement forward propagation
        pass

    def compute_loss(self, y_pred, y_true):
        # TO DO: Implement loss computation
        pass

    def backward_propagation(self, x, y, learning_rate):
        # TO DO: Implement backward propagation
        pass

    def train(self, x, y, epochs, learning_rate):
        #To DO: Implement training loop
        pass

    def predict(self, x):
        # TO DO: Implement prediction
        pass
