import numpy as np


class Dense:
    def __init__(self, 
                 n_inputs, n_neurons, 
                 weight_regularizer_l1=0., bias_regularizer_l1=0.,
                 weight_regularizer_l2=0., bias_regularizer_l2=0.):
        
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.output = np.dot(self.inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL = np.where(self.weights >= 0, 1, -1)
            self.dweights += self.weight_regularizer_l1 * dL
        
        if self.bias_regularizer_l1 > 0:
            dL = np.where(self.biases >= 0, 1, -1)
            self.dbiases += self.bias_regularizer_l1 * dL
        
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


class Dropout:
    def __init__(self, dropout_rate):
        # 'keep' rate for the layer
        self.rate = 1 - dropout_rate
        # for inference/prediction
        self.inference_mode = False
    
    def forward(self, inputs):
        self.inputs = inputs
        if not self.inference_mode:
            self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
            self.output = self.inputs * self.binary_mask
        else:
            self.output = inputs
    
    def backward(self, dvalues):
        if not self.inference_mode:
            self.dinputs = dvalues * self.binary_mask
        else:
            self.dinputs = dvalues
    
    def train(self):
        self.inference_mode = False
    
    def eval(self):
        self.inference_mode = True