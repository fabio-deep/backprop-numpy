import numpy as np
np.random.seed(8)

def initialize_weights(weight_init='he_normal'):

    if weight_init == 'glorot_uniform':
        W1 = glorot_uniform([512,784], 'sigmoid')
        W2 = glorot_uniform([512,512], 'sigmoid')
        W3 = glorot_uniform([10,512], 'sigmoid')

    elif weight_init == 'he_normal':
        W1 = he_normal([512,784])
        W2 = he_normal([512,512])
        W3 = he_normal([10,512])
    else:
        raise NotImplementedError

    b1 = np.zeros([512, 1])
    b2 = np.zeros([512, 1])
    b3 = np.zeros([10, 1])

    parameters = {'W1': W1,
                  'W2': W2,
                  'W3': W3,
                  'b1': b1,
                  'b2': b2,
                  'b3': b3}

    return parameters

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
