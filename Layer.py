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