import numpy as np

class Activation:
    def relu(x):
        return np.maximum(0, x)


class Layer:
    def __init__(self, dataDim: int, prevLayer, nextLayer):
        self.dataDim = dataDim
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer

    
class Input(Layer):
    def __init__(self, dataDim: int, prevLayer, nextLayer, data):
        super().__init__(dataDim, None, nextLayer)
        self.data = data


class Hidden(Layer):
    def __init__(self, dataDim: int, prevLayer, nextLayer, activation):
        super().__init__(dataDim, prevLayer, nextLayer)
        self.activation = activation
        self.weights = np.random.uniform(-1, 1, (prevLayer.dataDim, self.dataDim))
        self.biases = np.random.uniform(-1, 1, (self.dataDim, 1))

    
    def feedforward(self):
        z = np.dot(self.weights, self.prevLayer.data) + self.biases 

        if self.activation == "ReLU":
            return Activation.relu(z)
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
    def __init__(self, dataDim: int, prevLayer, nextLayer, activation):
        super().__init__(dataDim, prevLayer, None)
        self.activation = activation
        self.weights = np.random.uniform(-1, 1, (prevLayer.dataDim, self.dataDim))
        self.biases = np.random.uniform(-1, 1, (self.dataDim, 1))


    def feedforward(self):
        z = np.dot(self.weights, self.prevLayer.data) + self.biases 

        if self.activation == "ReLU":
            return Activation.relu(z) 
        else:
            return z