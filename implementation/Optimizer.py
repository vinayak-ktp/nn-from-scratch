import numpy as np


class Optimizer:
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr / (1 + self.decay * self.iterations)
    
    def post_update_params(self):
        self.iterations += 1


class SGD(Optimizer):
    def __init__(self, lr=1., decay=0., momentum=0.):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        layer.weight_momentums = self.momentum * layer.weight_momentums - self.current_lr * layer.dweights
        layer.bias_momentums = self.momentum * layer.bias_momentums - self.current_lr * layer.dbiases

        layer.weights += layer.weight_momentums
        layer.biases += layer.bias_momentums


class Adagrad(Optimizer):
    def __init__(self, lr=1., decay=0., epsilon=1e-7):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = layer.weight_cache + layer.dweights**2
        layer.bias_cache = layer.bias_cache + layer.dbiases**2

        layer.weights += -self.current_lr * (layer.dweights / np.sqrt(layer.weight_cache + self.epsilon))
        layer.biases += -self.current_lr * (layer.dbiases / np.sqrt(layer.bias_cache + self.epsilon))


class RMSprop(Optimizer):
    def __init__(self, lr=0.001, decay=0., rho=0.9, epsilon=1e-7):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.iterations = 0
        self.rho = rho
        self.epsilon = epsilon
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_lr * (layer.dweights / np.sqrt(layer.weight_cache + self.epsilon))
        layer.biases += -self.current_lr * (layer.dbiases / np.sqrt(layer.bias_cache + self.epsilon))


class Adam(Optimizer):
    def __init__(self, lr=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_momentum_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_lr * (weight_momentum_corrected / np.sqrt(weight_cache_corrected + self.epsilon))
        layer.biases += -self.current_lr * (bias_momentums_corrected / np.sqrt(bias_cache_corrected + self.epsilon))