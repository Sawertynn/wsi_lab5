import numpy as np


def laplace(x, mu=0, b=1):
    return (1 / (2 * b)) * np.exp(-np.abs(x - mu) / b)


class Perceptron_2_layers:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

        limit = 1 / np.sqrt(1)

        self.weights_first_layer = np.random.uniform(-limit, limit, (1, hidden_size))
        self.bias_first_layer = np.zeros((1, hidden_size))

        self.weights_second_layer = np.zeros((hidden_size, 1))
        self.bias_second_layer = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, x):
        # TO DO: Implement forward propagation
        pass

    def MSE(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward_propagation(self, x, y, learning_rate):
        # TO DO: Implement backward propagation
        pass

    def train(self, x, y, epochs, learning_rate):
        #To DO: Implement training loop
        pass

    def predict(self, x):
        # TO DO: Implement prediction
        pass
