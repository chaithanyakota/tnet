import numpy as np

def sigmoid(x):
    # f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred): 
    # y_true and y_pred are numpy arrays of same length
    return ((y_true - y_pred) ** 2).mean()

# class Neuron: 
#     def __init__(self, weights, bias):
#         self.weights = weights
#         self.bias = bias
    
#     def feedForward(self, inputs): 
#         total = np.dot(self.weights, inputs) + self.bias
#         return sigmoid(total)

class NeuralNetwork: 
    '''
        A neural network with:
            - 2 inputs
            - a hidden layer with 2 neurons (h1, h2)
            - an output layer with 1 neuron (o1)
        Each neuron has the same weights and bias:
            - w = [0, 1]
            - b = 0
    '''
    
    def __init__(self): 
        
        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        # biases 
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
     
    def feedForward(self, x):
        # x is a numpy array with 2 elements ( they're the inputs )
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
     
    def train(self, data, all_y_trues): 
        '''
        - data is a ( n x 2 ) numpy array, where n = # of samples in the dataset
        - all_y_trues is a numpy array with n elements
        - Elements in all_y_trues correspond with the elements in data
        '''
        
        learn_rate = 0.1
        epochs = 1000
        
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues): 
                # --- do a feedForward 
                h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
                h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
                o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
                
                y_pred = o1
                
                
                # --- calculating partial derivatives
                # --- naming: d_L_d_w1 represents "partial L / partial w1"
                
                d_L_d_ypred = -2 * (y_true - y_pred)
                
                # neuron h1
                # neuron h2
                # neuron o1
                
                
                
                