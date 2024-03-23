import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases


def relu(x):
    return np.maximum(x, 0)




