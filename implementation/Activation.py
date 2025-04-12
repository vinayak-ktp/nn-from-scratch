import numpy as np


class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output ** 2)


class Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        # sub the max output for each batch for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        exp_sum = np.sum(exp_values, axis=1, keepdims=True)
        self.output = exp_values / exp_sum
    
    def backward(self, dvalues):
        pass    # too complex, for later
