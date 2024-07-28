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
            self.data = Activation.relu(z)
        else:
            return z
    def backprop(self):
        y_hat = np.dot(self.weights, self.prevLayer.data) + self.biases #already calculated in feedforward
        rate = 0.001 #default value
        y = 0 #need expected output
        mse = 0.5 * (y_hat - y) ** 2
        deriv_yhat_loss = y_hat - y #gradient of mse with respect to y_hat
        deriv_yhat_w = np.outer(deriv_yhat_loss, self.prevLayer.data) #inputs(grad of y_hat with respect to w) * grad of mse respect to y_hat
        deriv_yhat_b = deriv_yhat_loss #1(grad of y_hat with respect to b) * grad of mse with respect to y_hat

        self.weights -= rate * deriv_yhat_w #w = w - rate * grad of loss with respect to w
        self.biases -= rate * deriv_yhat_b  #b = b- rate * grad of loss with respect to b


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
            self.data = Activation.relu(z)
        else:
            self.data = z 

        return self.data