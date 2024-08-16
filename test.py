import Layer
import numpy as np

# Test Feedfoward

def test_feed_forward():
    input = Layer.Input(3)
    hidden = Layer.Hidden(2, "ReLU")
    output = Layer.Output(1, "None")

    input.set_prev_next_layer(None, hidden)
    hidden.set_prev_next_layer(input, output)
    output.set_prev_next_layer(hidden, None)

    hidden.initialize()
    print(hidden.weights)
    print(hidden.biases)
    output.initialize()
    print(output.weights)
    print(output.biases)

    input.feed_forward(np.array([[1], [2], [3]]))
    hidden.feed_forward()
    print(output.feed_forward())


def test_backprop():
    input = Layer.Input(3, np.array([[1], [2], [3]]))
    hidden1 = Layer.Hidden(2, "ReLU")
    hidden2 = Layer.Hidden(2, "ReLU")
    output = Layer.Output(1, "None")

    input.set_prev_next_layer(None, hidden1)
    hidden1.set_prev_next_layer(input, hidden2)
    hidden2.set_prev_next_layer(hidden1, output)
    output.set_prev_next_layer(hidden2, None)

    hidden1.initialize()
    hidden2.initialize()
    output.initialize()

    hidden1.feed_forward()
    hidden2.feed_forward()
    print(output.feed_forward())

    print(output.backprop([[5]]))
    print(hidden2.backprop())
    print(hidden1.backprop())


def test_train():
    input = []
    for i in range(100):
        input.append([i])

    NN = Layer.NeuralNetwork("ReLU", 0, [1, 1], input, input)
    NN.train(learn_rate=0.00001, max_iter=1000)
    print(NN.test())

test_train()