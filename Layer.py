import numpy as np

class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def deriv_relu(x):
        # z = np.copy(x)
        # for i in range(z.shape[0]):
        #     for j in range(z.shape[1]):
        #         z[i][j] = 1 if z[i][j] > 0 else 0 

        # return z
        return np.where(x > 0, 1, 0)


class Layer:
    def __init__(self, data_dim: int):
        self.data_dim = data_dim

    
    def set_prev_next_layer(self, prev_layer, next_layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer

    
class Input(Layer):
    def __init__(self, data_dim: int):
        super().__init__(data_dim)
        

    def feed_forward(self, data):
        self.data = data

        return self.data
        


class Hidden(Layer):
    def __init__(self, data_dim: int, activation):
        super().__init__(data_dim)
        self.activation = activation


    def init_gradient(self):
        self.weights_gradient = np.zeros((self.data_dim, self.prev_layer.data_dim))
        self.biases_gradient = np.zeros((self.data_dim, 1))


    def initialize(self):
        self.weights = np.random.normal(0, (2 / self.data_dim) ** 0.5, (self.data_dim, self.prev_layer.data_dim))
        self.biases = np.zeros((self.data_dim, 1))

    
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
        else:
            self.delta = np.matmul(np.transpose(self.next_layer.weights), self.next_layer.delta)
            deriv_w = np.matmul(self.delta, np.transpose(self.prev_layer.data)) #inputs(grad of y_hat with respect to w) * grad of mse respect to y_hat
            deriv_b = self.delta #1(grad of y_hat with respect to b) * grad of mse with respect to y_hat

            return deriv_w, deriv_b
        


class Output(Layer):
    def __init__(self, data_dim: int, activation):
        super().__init__(data_dim)
        self.activation = activation

    def init_gradient(self):
        self.weights_gradient = np.zeros((self.data_dim, self.prev_layer.data_dim))
        self.biases_gradient = np.zeros((self.data_dim, 1))

     
    def initialize(self):
        self.weights = np.random.normal(0, (2 / self.data_dim) ** 0.5, (self.data_dim, self.prev_layer.data_dim))
        self.biases = np.zeros((self.data_dim, 1))


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
    

class NeuralNetwork:
    def  __init__(self, activation, hidden: int, layer_dimensions: list, x, y):
        if hidden + 2 != len(layer_dimensions):
            raise Exception("Invalid Arguments")
        else:
            self.activation = activation
            self.hidden = hidden
            self.layer_dimensions = layer_dimensions
            self.x = x
            self.y = y
            self.neural_network = []
            input = Input(layer_dimensions[0])
            self.neural_network.append(input)

            for i in range(1, len(layer_dimensions) - 1):
                hidden = Hidden(layer_dimensions[i], activation)
                self.neural_network.append(hidden)

            output = Output(layer_dimensions[-1], "None")
            self.neural_network.append(output)

            self.neural_network[0].set_prev_next_layer(None, self.neural_network[1])
            
            for i in range(1, len(self.neural_network) - 1):
                self.neural_network[i].set_prev_next_layer(self.neural_network[i - 1], self.neural_network[i + 1])

            self.neural_network[-1].set_prev_next_layer(self.neural_network[-2], None)

            for i in range(1, len(self.neural_network)):
                self.neural_network[i].initialize()

            # self.x_train = self.x[:int(.8 * len(x))]
            # self.y_train = self.y[:int(.8 * len(y))]
            # self.x_test = self.x[-int(.2 * len(x)):]
            # self.y_test = self.y[-int(.2 * len(y)):]
            self.x_train = x
            self.y_train = y
            self.x_test = x
            self.y_test = y


    def feed_forward(self, data):
        self.neural_network[0].feed_forward(data)
        for i in range(1, len(self.neural_network) - 1):
            self.neural_network[i].feed_forward()

        return self.neural_network[-1].feed_forward()
    

    def train(self, learn_rate = 0.001, max_iter = 1000):
        for i in range(1, max_iter + 1):
            for j in range(1, len(self.neural_network)):
                self.neural_network[j].init_gradient()

            total_loss = 0

            for k in range(len(self.x_train)):
                x = np.transpose(np.atleast_2d(self.x_train[k]))
                y_hat = self.feed_forward(x)
                y = np.transpose(np.atleast_2d(self.y_train[k]))

                total_loss += (y_hat - y)**2

                call_backprop = self.neural_network[-1].backprop(y)
                self.neural_network[-1].weights_gradient += call_backprop[0]
                self.neural_network[-1].biases_gradient += call_backprop[1]

                for m in range(2, len(self.neural_network)):
                    call_backprop = self.neural_network[-m].backprop()
                    self.neural_network[-m].weights_gradient += call_backprop[0]
                    self.neural_network[-m].biases_gradient += call_backprop[1]
                    
            avg_loss = total_loss / (2 * len(self.x_train))

            for n in range(1, len(self.neural_network)):
                self.neural_network[n].weights -= learn_rate * self.neural_network[n].weights_gradient
                self.neural_network[n].biases -= learn_rate * self.neural_network[n].biases_gradient

            print(f"Iteration {i}/{max_iter}, Loss: {avg_loss}")
        

    def test(self):
        loss = 0

        for i in range(0, len(self.x_test)):
            y_hat = self.feed_forward(np.transpose(np.atleast_2d(self.x_test[i])))
            y = self.y_test[i]
            print(y_hat, y)
            loss += (y_hat - y)**2

        mse = loss / (2 * len(self.x_test))
        return mse
    
