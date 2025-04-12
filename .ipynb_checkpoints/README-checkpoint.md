# Neural Network From Scratch

This repository contains the implementation of a neural network from scratch using Python and NumPy. The neural network is designed to handle classification tasks, using a variety of optimizers and activation functions. The project is an educational exercise to understand the inner workings of neural networks, including the backpropagation algorithm, and to implement various components such as activation functions, loss functions, and optimizers.

## Features

- **Neural Network Architecture**: A fully connected feedforward neural network.
- **Activation Functions**: ReLU, Tanh, Sigmoid and Softmax activation functions.
- **Optimizers**: Implemented common optimizers such as SGD, Adam, RMSprop, and Adagrad from scratch.
- **Loss Functions**: Categorical Cross-Entropy, Binary Cross-Entropy and Mean Squared Error.
- **Regularization**: L1 and L2 regularization and dropout to prevent overfitting and dropout to improve generalization.

## Current Implementations

### 1. **Neural Network Layers**
   - **Dense Layer**: Fully connected layer with weights and biases.
   - **Activation Layer**: ReLU and Softmax activation functions.
   
### 2. **Optimizers**
   - **SGD (Stochastic Gradient Descent)**: Basic optimizer with momentum.
   - **Adam**: Adaptive learning rate with momentum and bias correction.
   - **RMSprop**: Optimizer with exponentially decaying averages of squared gradients.
   - **Adagrad**: Adaptive learning rate based on past gradients.

### 3. **Loss Functions**
   - **Categorical Cross-Entropy**: Used for multi-class classification tasks.
   - **Mean Squared Error (MSE)**: Used for regression tasks.

### 4. **Regularization**

- **L1 Regularization**: Adds a penalty equal to the absolute value of the weights to the loss function. It encourages sparsity in the model.
  
- **L2 Regularization**: Adds a penalty equal to the square of the weights to the loss function. It helps prevent the model from overfitting by discouraging large weights.

- **Dropout**: A technique where, during training, random units in the network are set to zero with a certain probability. This helps in reducing overfitting and improves the model's ability to generalize.

## Structure
```bash
nn-from-scratch
├─ implementation       # segmented code
│  ├─ Layer.py 
│  ├─ Activation.py
│  ├─ Loss.py
│  ├─ Optimizer.py
│  └─ test.ipynb
└─ spelled_out.ipynb    # entirety of code
```

### **Feel free to fork the repo and open pull requests!**