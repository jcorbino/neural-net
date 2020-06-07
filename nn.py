import numpy as np

# Equivalent to calling SciPy's signal.correlate2d(image, filter, mode = 'same')
def conv2d(image, filter, mode = 'same', pad_val = 0):
    if mode == 'same':
        image = np.pad(image, 1, constant_values = pad_val)
    # shape = (shape of filter, shape of output)
    shape = filter.shape + tuple(np.subtract(image.shape, filter.shape) + 1)
    # Subarrays formed by moving the "window" row-wise by a stride of 1
    sub = np.lib.stride_tricks.as_strided(image, shape, image.strides*2)
    return np.einsum('ij, ijkl -> kl', filter, sub)

# Use NumPy's polyfit function instead
def linear_regression(X, y, Q = None):
    bias = np.ones((X.shape[0], 1))
    X = np.append(bias, X, axis = 1)
    # Normal equation
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    if Q is not None:
        return theta[1]*Q + theta[0]
    else:
        return theta[1]*X[:, 1:] + theta[0]

# Initializers
def rand(nrows, ncols, which):
    return {
        'rand' : np.random.rand(nrows, ncols),
        'uniform' : np.random.uniform(-1, 1, (nrows, ncols)),
        'normal' : np.random.normal(0, 1, (nrows, ncols))
    }[which]

# Activation functions
def activation(x, which):
    return {
        'sigmoid' : 1/(1 + np.exp(-x)),
        'tanh' : (np.exp(2*x) - 1)/(np.exp(2*x) + 1),
        'relu' : np.maximum(0, x)
    }[which]

def activation_derivative(x, which):
    return {
        'sigmoid' : x*(1 - x),
        'tanh' : 1 - np.square(x),
        'relu' : np.greater(x, 0).astype(int)
    }[which]

class NeuralNetwork:
    # Constructor only initializes weights and biases, anything else can be provided at training time
    def __init__(self, input_len, hiddens_len, output_len, initializer = 'uniform'):
        self.weights = [rand(input_len, hiddens_len[0], initializer)*pow(input_len, -0.5)]
        self.biases = [np.zeros((1, hiddens_len[0]))]
        for i in range(1, len(hiddens_len)):
            self.weights.append(rand(hiddens_len[i-1], hiddens_len[i], initializer)*pow(hiddens_len[i-1], -0.5))
            self.biases.append(np.zeros((1, hiddens_len[i])))
        self.weights.append(rand(hiddens_len[-1], output_len, initializer)*pow(hiddens_len[-1], -0.5))
        self.biases.append(np.zeros((1, output_len)))

    def __feedforward(self, inputs):
        self.outputs = [inputs]
        for w, b, a in zip(self.weights, self.biases, self.activations):
            self.outputs.append(activation(self.outputs[-1] @ w + b, a))

    def __backprop(self, outputs, lr):
        error = outputs - self.outputs[-1]
        for i in range(len(self.weights)-1, -1, -1):
            delta = lr*error*activation_derivative(self.outputs.pop(), self.activations[i])
            # Propagate error backwards
            error = error @ self.weights[i].T
            # Update weights
            self.weights[i] += self.outputs[-1].T @ delta
            # Update biases
            self.biases[i] += delta.sum(axis = 0)

    def train(self, inputs, outputs, activations = [], epochs = 100, lr = 0.001, batch_size = 0):
        if not activations:
            self.activations = ['relu']*len(self.weights)
        else:
            self.activations = activations
        if not batch_size:
            # Default: Gradient descent (full vectorization)
            batch_size = inputs.shape[0]
        for _ in range(epochs):
            print('Epoch', _)
            # Shuffle training data (recommended)
            idx = np.random.permutation(inputs.shape[0])
            for i in range(0, inputs.shape[0], batch_size):
                self.__feedforward(inputs[idx][i:i + batch_size])
                self.__backprop(outputs[idx][i:i + batch_size], lr)

    def predict(self, inputs):
        self.__feedforward(inputs)
        return self.outputs[-1]
