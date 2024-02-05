import numpy as np

def sigmoid(x):
    # f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron: 
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedForward(self, inputs): 
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
