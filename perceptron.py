import numpy as np
import matplotlib.pyplot as plt


def laplace(x, mu=0, b=1):
    return (1 / (2 * b)) * np.exp(-np.abs(x - mu) / b)

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

class Perceptron_2_layers:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

        limit = 1 / np.sqrt(1)

        self.weights_first_layer = np.random.uniform(-limit, limit,
                                                     (1, hidden_size))
        self.bias_first_layer = np.zeros((1, hidden_size))

        self.weights_second_layer = np.zeros((hidden_size, 1))
        self.bias_second_layer = np.zeros((1, 1))

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, x):
        self.hidden_layer = (np.dot(x, self.weights_first_layer) +
                             self.bias_first_layer)
        self.hidden_layer = self.sigmoid(self.hidden_layer)
        self.output = (np.dot(self.hidden_layer, self.weights_second_layer) +
                       self.bias_second_layer)

        return self.output


    def backward_propagation(self, x, y, learning_rate):
        output_error = self.output - y
        delta_output = output_error

        delta_hidden = (np.dot(delta_output, self.weights_second_layer.T) *
                        self.sigmoid_derivative(self.hidden_layer))
        self.weights_second_layer -= (np.dot(self.hidden_layer.T,
                                             delta_output) * learning_rate)
        self.bias_second_layer -= (np.sum(delta_output, axis=0,
                                          keepdims=True) * learning_rate)

        self.weights_first_layer -= np.dot(x.T, delta_hidden) * learning_rate
        self.bias_first_layer -= (np.sum(delta_hidden, axis=0, keepdims=True)
                                  * learning_rate)
        
    def train_unnormalized(self, x, y, epochs, learning_rate, printing=True):
        for epoch in range(epochs):
            self.forward_propagation(x)
            self.backward_propagation(x, y, learning_rate)
            if printing and (epoch % 100) == 0:
                mse = MSE(y, self.output)
                mae = MAE(y, self.output)
                print(f'Epoch {epoch}, MSE: {mse:.2e}, MAE: {mae:.2e}')
    
    def predict_unnormalized(self, x):
        return self.forward_propagation(x)

    def train(self, x, y, epochs, learning_rate, printing=True):
        # Normalize the input data
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        x_norm = (x - self.x_mean) / self.x_std

        # Normalize the output data
        self.y_mean = np.mean(y, axis=0)
        self.y_std = np.std(y, axis=0)
        y_norm = (y - self.y_mean) / self.y_std

        for epoch in range(epochs):
            self.forward_propagation(x_norm)
            self.backward_propagation(x_norm, y_norm, learning_rate)
            if printing and (epoch % 1000) == 0:
                mse = MSE(y_norm, self.output)
                mae = MAE(y_norm, self.output)
                print(f'Epoch {epoch}, MSE: {mse:.2e}, MAE: {mae:.2e}')

    def predict(self, x):
        x_norm = (x - self.x_mean) / self.x_std
        y_pred_norm = self.forward_propagation(x_norm)

        y_pred = y_pred_norm * self.y_std + self.y_mean
        return y_pred


if __name__ == "__main__":
    x = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = laplace(x).reshape(-1, 1)

    hidden_size = 5
    learning_rate = 0.01
    epochs = 100_000
    model = Perceptron_2_layers(hidden_size)
    model.train(x, y, epochs, learning_rate)

    y_pred = model.predict(x)

    # plot the results
    plt.plot(x, y, label='True')
    plt.plot(x, y_pred, label='Predicted')
    plt.legend()
    plt.show()
