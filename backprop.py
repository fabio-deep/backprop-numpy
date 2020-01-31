import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
np.random.seed(8)
tf.set_random_seed(8)
epsilon = 1e-7

def sigmoid(z):
    #z = 1. / (1. + np.exp(-z))+epsilon
    return .5 * (1 + np.tanh(.5*z))

def relu(z):
    z[z<0] = 0
    return z

def tanh(z):
    numerator = np.exp(z) - np.exp(-z)
    denominator = np.exp(z) + np.exp(-z)
    z = numerator / denominator+epsilon
    return z

def softmax(z):
    softmax = np.zeros_like(z)
    for i in range(z.shape[1]):
        z[:,i] -= np.max(z[:,i]) # max subtraction for stability
        softmax[:,i] = np.exp(z[:,i]) / np.sum(np.exp(z[:,i]))+epsilon
    return softmax

# def derivative_softmax(yhat):
#     yhat = softmax(yhat)
#     jacobian = np.zeros([10,10,32])
#     for m in range(yhat.shape[1]):
#         J = np.diag(yhat[:,m])
#         for i in range(len(jacobian)):
#             for j in range(len(jacobian)):
#                 if i == j:
#                     J[i][j] = yhat[:,m][i] * (1-yhat[:,m][i]) # yhat_i * (1-yhat_i)
#                 else:
#                     J[i][j] = -yhat[:,m][i] * yhat[:,m][j] # -yhat_i*yhat_j
#         jacobian[:,:,m] = J
#
#     return jacobian.diagonal().T

def derivative_softmax(yhat):
    yhat = softmax(yhat).T
    yhat_diag = np.zeros((yhat.shape[0], yhat.shape[1], yhat.shape[1]))
    diag_idx = np.arange(yhat.shape[1])
    yhat_diag[:,diag_idx, diag_idx] = yhat
    jacobiana = (yhat_diag - np.expand_dims(yhat,-1) @ np.expand_dims(yhat, 1)).T

    return jacobiana.diagonal().T

    # J = - yhat[:,:, None] * yhat[:, None, :] # off-diagonal Jacobian
    # iy, ix = np.diag_indices_from(J[0])
    # J[:, iy, ix] = yhat * (1. - yhat) # diagonal
    # return J.T.diagonal().T

def derivative_sigmoid(z):
    d_sigmoid = sigmoid(z) * (1. - sigmoid(z))
    return d_sigmoid

def derivative_tanh(z):
    d_tanh = 1. - tanh(z)**2.
    return d_tanh

def derivative_relu(z):
     z[z<=0] = 0
     z[z>0] = 1
     return z

def binary_cross_entropy(y_hat, y):
    cost = np.mean(-(y*np.log(y_hat) + (1. - y)*np.log(1. - y_hat)))
    return cost

def categorical_cross_entropy(y_hat, y):
    cost = np.mean(-np.sum(y*np.log(y_hat), 1))
    return cost

def glorot_uniform(shape, activation):
    synapse_in = shape[0]
    synapse_out = shape[1]
    c = 4 if activation == 'sigmoid' else 1 # {sigmoid: 4, tanh: 1}

    weights = np.random.uniform(low = c*-np.sqrt(6 / (synapse_in + synapse_out)),
                                high = c*np.sqrt(6 / (synapse_in + synapse_out)),
                                size = (synapse_in, synapse_out))
    return weights

def he_normal(shape):
    synapse_in = shape[0]
    synapse_out = shape[1]
    weights = np.random.randn(synapse_in, synapse_out)*np.sqrt(2./synapse_in)
    return weights

def initialize_weights():
    #W1 = glorot_uniform([256,784], 'sigmoid')
    W1 = he_normal([256,784])
    b1 = np.zeros([256, 1])
    #W2 = glorot_uniform([256,256], 'sigmoid')
    W2 = he_normal([256,256])
    b2 = np.zeros([256, 1])
    #W3 = glorot_uniform([10,256], 'sigmoid')
    W3 = he_normal([10,256])
    b3 = np.zeros([10, 1])

    parameters = {'W1': W1,
                  'W2': W2,
                  'W3': W3,
                  'b1': b1,
                  'b2': b2,
                  'b3': b3}

    return parameters

def forward_propagation(A0, parameters, batch_size):

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']

    Z1 = W1 @ A0.T + b1
    A1 = relu(Z1)

    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)

    Z3 = W3 @ A2 + b3
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

def adam_optimiser(parameters, gradients, alpha):
    return parameters

def train_model(n_epochs, batch_size=32, alpha=0.001):

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    train_steps_per_epoch = int(np.ceil(mnist.train.num_examples / batch_size))
    valid_steps_per_epoch = int(np.ceil(mnist.validation.num_examples / batch_size))
    test_steps_per_epoch = int(np.ceil(mnist.test.num_examples / batch_size))

    train_cost = 0.
    train_accuracy = 0.
    test_cost = 0.
    test_accuracy = 0.
    parameters = initialize_weights()

    for epoch in range(n_epochs):
        valid_cost = 0.
        valid_accuracy = 0.
        for iter in range(train_steps_per_epoch):

            train_X, train_Y = mnist.train.next_batch(batch_size)

            activations = forward_propagation(train_X, parameters, batch_size)
            train_cost += categorical_cross_entropy(activations['A3'], train_Y.T)
            train_accuracy += np.mean(np.argmax(activations['A3'].T,1) == np.argmax(train_Y, 1))
            gradients = back_propagation(train_X, activations, parameters, train_Y)
            parameters = sgd_optimiser(parameters, gradients, alpha)

            if iter % 100==0:
                print('Iter: {}/{} || Loss: {:.4f} || Accuracy: {:.4f}'.format(iter+1,
                    train_steps_per_epoch,
                    train_cost/(epoch*train_steps_per_epoch+(iter+1)),
                    train_accuracy/(epoch*train_steps_per_epoch+(iter+1))))

        for i in range(valid_steps_per_epoch):
            valid_X, valid_Y = mnist.validation.next_batch(batch_size)

            activations = forward_propagation(valid_X, parameters, batch_size)
            valid_cost += categorical_cross_entropy(activations['A3'], valid_Y.T) / valid_steps_per_epoch
            valid_accuracy += np.mean(np.argmax(activations['A3'].T,1) == np.argmax(valid_Y, 1)) / valid_steps_per_epoch

        print('\nEpoch: {}/{} Validation: || Loss: {:.4f} || Accuracy: {:.4f}'.format(epoch+1, n_epochs, valid_cost, valid_accuracy),'\n')

    for i in range(test_steps_per_epoch):
        test_X, test_Y = mnist.test.next_batch(batch_size)

        activations = forward_propagation(test_X, parameters, batch_size)
        test_cost += categorical_cross_entropy(activations['A3'], test_Y.T) / test_steps_per_epoch
        test_accuracy += np.mean(np.argmax(activations['A3'].T,1) == np.argmax(test_Y, 1)) / test_steps_per_epoch

    print('Test Results: || Loss {:.4f} || Accuracy {:.4f}'.format(test_cost, test_accuracy))

    return test_cost, test_accuracy

test_cost, test_accuracy = train_model(n_epochs=20, batch_size=32, alpha=.01)
