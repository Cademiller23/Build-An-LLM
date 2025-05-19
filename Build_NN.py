
"""
Conventional neural networks, all nodes in these layers
are interconnected to form a dense network.

Certain nodes in the network are not connected to each other,
these are called sparse networks.
InceptinNet models for image classification use sparse networks.
FeedForward - process in which inputs will be passed through
neuron to the next layer after activation.

Activation function is a function which decides whether 
a neuron needs to be activated or not.
"""

import numpy as np 

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights 
        self.bias = bias
    
    def feedforward(self, x):
        return np.dot(weights, x) + self.bias
    
weights = np.array([0,1])
bias = 1
n = Neuron(weights, bias)
print(n.feedforward([1,1]))

# Sigmoid function sig(t) = 1/(1+e^-t)
# softmax activation function soft(t) = e^t/sum(e^t)
def sigmoid(input):
    return 1/(1+np.exp(input))

def relu(input):
    return max(0, input)

def softmax(input):
    return np.exp(input)/np.sum(np.exp(input))
