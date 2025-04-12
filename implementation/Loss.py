import numpy as np
from Activation import Softmax, Sigmoid


class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        data_loss = np.mean(sample_losses)
        return data_loss

 
class CCELoss(Loss):
    def forward(self, y_pred, y_true):
        # to avoid log of 0 or 1
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # get the correct class indices
        if(len(y_true.shape) > 1):
            y_true = np.argmax(y_true, axis=1)
        # pick predicted values at correct class indices
        pred_vals = y_pred[range(len(y_true)), y_true]
        # find the per sample likelihood/loss
        neg_log_likelihoods = -np.log(pred_vals)
        return neg_log_likelihoods
    
    def backward(self, y_pred, y_true):
        n_samples = len(y_true)
        n_labels = len(y_true[0])
        # one-hot encode the values
        if(len(y_true.shape) == 1):
            y_true = np.eye(n_labels)[y_true]
        # calculate gradients (normalized)
        self.dinputs = - (y_true / y_pred) / n_samples


class CategoricalCrossEntropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CCELoss()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output  # y_pred
        self.loss_value = self.loss.calculate(self.output, y_true)

    def backward(self, y_pred, y_true):
        n_samples = len(y_pred)
        n_labels = len(y_pred[0])
        # one-hot encode the values
        if(len(y_true.shape) == 1):
            y_true = np.eye(n_labels)[y_true]
        # calculate gradient (normalized)
        self.dinputs = (y_pred - y_true) / n_samples


class BCELoss(Loss):
    def forward(self, y_pred, y_true):
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # reshape if necessary
        if(len(y_pred.shape) == 1):
            y_pred = y_pred.reshape(-1, 1)
        if(len(y_true.shape) == 1):
            y_true = y_true.reshape(-1, 1)
        # calculate per sample losses
        neg_log_likelihood = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return neg_log_likelihood
    
    def backward(self, y_pred, y_true):
        n_samples = len(y_true)
        # reshape if necessary
        if(len(y_pred.shape) == 1):
            y_pred = y_pred.reshape(-1, 1)
        if(len(y_true.shape) == 1):
            y_true = y_true.reshape(-1, 1)
        # calculate (normalized) gradients
        self.dinputs = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n_samples


class BinaryCrossEntropy:
    def __init__(self):
        self.activation = Sigmoid()
        self.loss = BCELoss()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        self.loss_value = self.loss.calculate(self.output, y_true)
    
    def backward(self, y_true, y_pred):
        n_samples = len(y_true)
        # reshpae if necessary
        if(len(y_pred.shape) == 1):
            y_pred = y_pred.reshape(-1, 1)
        if(len(y_true.shape) == 1):
            y_true = y_true.reshape(-1, 1)
        # calculate (normalized) gradients
        self.dinputs = (y_pred - y_true) / n_samples


class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        squared_error = np.square(y_pred - y_true)
        return np.mean(squared_error)
    
    def backward(self, y_pred, y_true):
        n_samples = len(y_true)
        # reshape to [n, 1] for 1d regression 
        if(len(y_true.shape) == 1):
            y_true = y_true.reshape(-1, 1)
        # normalized gradient
        self.dinputs = (2 * (y_pred - y_true)) / n_samples