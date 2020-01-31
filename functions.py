import numpy as np
epsilon = 1e-9

def sigmoid(z):
    #z = 1. / (1. + np.exp(-z))+epsilon
    return .5 * (1 + np.tanh(.5*z))

def derivative_sigmoid(z):
    d_sigmoid = sigmoid(z) * (1. - sigmoid(z))
    return d_sigmoid

def relu(z):
    z[z<0] = 0
    return z

def derivative_relu(z):
     z[z<=0] = 0
     z[z>0] = 1
     return z

def tanh(z):
    numerator = np.exp(z) - np.exp(-z)
    denominator = np.exp(z) + np.exp(-z)
    z = numerator / denominator+epsilon
    return z

def derivative_tanh(z):
    d_tanh = 1. - tanh(z)**2.
    return d_tanh

def softmax(z):
    softmax = np.zeros_like(z)
    for i in range(z.shape[1]):
        z[:,i] -= np.max(z[:,i]) # max subtraction for stability
        softmax[:,i] = np.exp(z[:,i]) / np.sum(np.exp(z[:,i]))+epsilon
    return softmax

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
