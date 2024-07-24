import numpy as np

class Activation:
    def relu(x):
        return np.maximum(0, x)


class Layer:
    def __init__(self, data_dim: int):
        self.data_dim = data_dim

    
    def set_prev_next_layer(self, prev_layer, next_layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer

    
class Input(Layer):
    def __init__(self, data_dim: int, data):
        super().__init__(data_dim)
        self.data = data


class Hidden(Layer):
    def __init__(self, data_dim: int, activation):
        super().__init__(data_dim)
        self.activation = activation


    def initialize(self):
        self.weights = np.random.uniform(-1, 1, (self.data_dim, self.prev_layer.data_dim))
        self.biases = np.random.uniform(-1, 1, (self.data_dim, 1))

    
    def feed_forward(self):
        z = np.dot(self.weights, self.prev_layer.data) + self.biases 

        if self.activation == "ReLU":
            return Activation.relu(z)
        else:
            return z


class Output(Layer):
    def __init__(self, data_dim: int, activation):
        super().__init__(data_dim)
        self.activation = activation

     
    def initialize(self):
        self.weights = np.random.uniform(-1, 1, (self.data_dim, self.prev_layer.data_dim))
        self.biases = np.random.uniform(-1, 1, (self.data_dim, 1))


    def feed_forward(self):
        z = np.dot(self.weights, self.prev_layer.data) + self.biases 

        if self.activation == "ReLU":
            return Activation.relu(z) 
        else:
            return z