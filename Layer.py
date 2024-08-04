import numpy as np

class Activation:
    def relu(x):
        return np.maximum(0, x)
    

    def deriv_relu(x):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i][j] = 1 if x[i][j] > 0 else 0 

        return x


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
        self.z = np.dot(self.weights, self.prev_layer.data) + self.biases 

        if self.activation == "ReLU":
            self.data = Activation.relu(self.z)
        else:
            self.data = self.z
        
        return self.data
        

    def backprop(self):
        if self.activation == "ReLU":
            self.delta = np.matmul(np.transpose(self.next_layer.weights), self.next_layer.delta) * Activation.deriv_relu(self.z)
            deriv_w = np.matmul(self.delta, np.transpose(self.prev_layer.data)) #inputs(grad of y_hat with respect to w) * grad of mse respect to y_hat
            deriv_b = self.delta #1(grad of y_hat with respect to b) * grad of mse with respect to y_hat

            return deriv_w, deriv_b


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
    

    def backprop(self, y):
        y_hat = self.data
        deriv_yhat_loss = (y_hat - y) / self.data_dim #gradient of mse with respect to y_hat
        deriv_w = np.matmul(deriv_yhat_loss, np.transpose(self.prev_layer.data)) #inputs(grad of y_hat with respect to w) * grad of mse respect to y_hat
        deriv_b = deriv_yhat_loss #1(grad of y_hat with respect to b) * grad of mse with respect to y_hat
        self.delta = deriv_yhat_loss

        return deriv_w, deriv_b