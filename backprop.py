import numpy as np

from functions import *

def forward_propagation(A0, parameters, batch_size):

    Z1 = parameters['W1'] @ A0.T + parameters['b1']
    A1 = relu(Z1)

    Z2 = parameters['W2'] @ A1 + parameters['b2']
    A2 = relu(Z2)

    Z3 = parameters['W3'] @ A2 + parameters['b3']
    A3 = softmax(Z3)

    activations = { 'Z1': Z1,
                    'A1': A1,
                    'Z2': Z2,
                    'A2': A2,
                    'Z3': Z3,
                    'A3': A3}
    return activations

def back_propagation(A0, activations, parameters, Y):

    Z1 = activations['Z1']
    A1 = activations['A1']
    Z2 = activations['Z2']
    A2 = activations['A2']
    Z3 = activations['Z3']
    A3 = activations['A3']

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    N = A0.shape[0]

    dA3 = -(Y.T)/A3 + (1-Y.T)/(1-A3) # ∂E/∂A3
    dZ3 = dA3 * derivative_softmax(Z3) # ∂E/∂A3 * ∂A3/∂Z3, same as A3-Y
    dW3 = A2 # ∂Z3/∂W3
    jacobian_W3 = (1./N) * dZ3 @ dW3.T # ∂E/∂A3 * ∂A3/∂Z3 @ ∂Z3/∂W3 = ∂E/∂W3
    db3 = 1. # ∂Z3/∂b3
    jacobian_b3 = (1./N) * np.sum(dZ3 * db3, axis=1, keepdims=True) # ∂E/∂A3 * ∂A3/∂Z3 * ∂Z3/∂b3 = ∂E/∂b3

    dA2 = W3.T @ dZ3 # ∂E/∂A3 * ∂A3/∂Z3 @ ∂Z3/∂A2
    dZ2 = dA2 * derivative_relu(Z2) # ∂E/∂A3 * ∂A3/∂Z3 @ ∂Z3/∂A2 * ∂A2/∂Z2
    dW2 = A1 # ∂Z2/∂W2
    jacobian_W2 = (1./N) * dZ2 @ dW2.T # ∂E/∂A3 * ∂A3/∂Z3 @ ∂Z3/∂A2 * ∂A2/∂Z2 @ ∂Z2/∂W2 = ∂E/∂W2
    db2 = 1. # ∂Z2/∂b2
    jacobian_b2 = (1/N) * np.sum(dZ2 * db2, axis=1, keepdims=True) # ∂E/∂A3 * ∂A3/∂Z3 @ ∂Z3/∂A2 * ∂A2/∂Z2 @ ∂Z2/∂b2 = ∂E/∂b2

    dA1 = W2.T @ dZ2 # ∂E/∂A3 * ∂A3/∂Z3 @ ∂Z3/∂A2 * ∂A2/∂Z2 @ ∂Z2/∂A1
    dZ1 = dA1 * derivative_relu(Z1) # ∂E/∂A3 * ∂A3/∂Z3 @ ∂Z3/∂A2 * ∂A2/∂Z2 @ ∂Z2/∂A1 * ∂A1/∂Z1
    dW1 = A0 # ∂Z1/∂W1
    jacobian_W1 = (1./N) * dZ1 @ dW1 # ∂E/∂A3 * ∂A3/∂Z3 @ ∂Z3/∂A2 * ∂A2/∂Z2 @ ∂Z2/∂A1 * ∂A1/∂Z1 @ ∂Z1/∂W1 = ∂E/∂W1
    db1 = 1. # ∂Z1/∂b1
    jacobian_b1 = (1./N) * np.sum(dZ1 * db1, axis=1, keepdims=True) # ∂E/∂A3 * ∂A3/∂Z3 @ ∂Z3/∂A2 * ∂A2/∂Z2 @ ∂Z2/∂A1 * ∂A1/∂Z1 @ ∂Z1/∂b1 = ∂E/∂b1

    gradients = {'dW3' : jacobian_W3,
                 'dW2' : jacobian_W2,
                 'dW1' : jacobian_W1,
                 'db3' : jacobian_b3,
                 'db2' : jacobian_b2,
                 'db1' : jacobian_b1}

    return gradients

def sgd_optimiser(parameters, gradients, alpha):

    parameters['W3'] = parameters['W3'] - alpha*gradients['dW3']
    parameters['W2'] = parameters['W2'] - alpha*gradients['dW2']
    parameters['W1'] = parameters['W1'] - alpha*gradients['dW1']
    parameters['b3'] = parameters['b3'] - alpha*gradients['db3']
    parameters['b2'] = parameters['b2'] - alpha*gradients['db2']
    parameters['b1'] = parameters['b1'] - alpha*gradients['db1']

    return parameters
