import Layer
import numpy as np

# Test Feedfoward

def test_feed_forward():
    input = Layer.Input(3, np.array([[1], [2], [3]]))
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

    hidden.feed_forward()
    print(output.feed_forward())


test_feed_forward()