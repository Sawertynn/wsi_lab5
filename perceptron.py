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

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, x):
        self.hidden_layer = np.dot(x, self.weights_first_layer) + self.bias_first_layer
        self.hidden_layer = self.sigmoid(self.hidden_layer)
        self.output = np.dot(self.hidden_layer, self.weights_second_layer) + self.bias_second_layer

        return self.output

    def MSE(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward_propagation(self, x, y, learning_rate):
        output_error = self.output - y
        delta_output = output_error  

        delta_hidden = np.dot(delta_output, self.weights_second_layer.T) * self.sigmoid_derivative(self.hidden_layer)
        self.weights_second_layer -= np.dot(self.hidden_layer.T, delta_output) * learning_rate
        self.bias_second_layer -= np.sum(delta_output, axis=0, keepdims=True) * learning_rate

        self.weights_first_layer -= np.dot(x.T, delta_hidden) * learning_rate
        self.bias_first_layer -= np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward_propagation(x)
            self.backward_propagation(x, y, learning_rate)
            if (epoch % 100) == 0:
                mse = self.MSE(y,self.output)
                print(f'Epoch {epoch}, MSE: {mse}')

    def predict(self, x):
        return self.forward_propagation(x)
