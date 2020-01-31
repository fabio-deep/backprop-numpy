import numpy as np

from weight_init import *
from backprop import *
from losses import *

def train(mnist, n_epochs, batch_size, alpha):

    train_steps_per_epoch = int(np.ceil(mnist.train.num_examples / batch_size))
    valid_steps_per_epoch = int(np.ceil(mnist.validation.num_examples / batch_size))
    test_steps_per_epoch = int(np.ceil(mnist.test.num_examples / batch_size))

    parameters = initialize_weights('he_normal')
    print('\nStarting training.. \n')

    for epoch in range(n_epochs):
        # train
        sample_count = 0.
        train_cost = 0.
        train_accuracy = 0.

        for iter in range(train_steps_per_epoch):
            train_X, train_Y = mnist.train.next_batch(batch_size)
            sample_count += int(train_X.shape[0])

            activations = forward_propagation(train_X, parameters, batch_size)

            train_cost += categorical_cross_entropy(activations['A3'], train_Y.T) * int(train_X.shape[0])
            train_accuracy += np.sum(np.argmax(activations['A3'].T,1) == np.argmax(train_Y, 1))

            gradients = back_propagation(train_X, activations, parameters, train_Y)
            parameters = sgd_optimiser(parameters, gradients, alpha)

        epoch_train_loss = train_cost / sample_count
        epoch_train_acc = train_accuracy / sample_count

        # validation
        epoch_valid_loss, epoch_valid_acc = evaluate(mnist.validation,
            parameters, valid_steps_per_epoch, batch_size)

        print('\nEpoch: {}/{} - Train loss: {:.4f} - acc: {:.4f} | Valid loss: {:.4f} - acc: {:.4f}'.format(
            epoch+1, n_epochs,
            epoch_train_loss, epoch_train_acc,
            epoch_valid_loss, epoch_valid_acc),'\n')

    # test
    epoch_test_loss, epoch_test_acc = evaluate(mnist.test,
        parameters, test_steps_per_epoch, batch_size)

    print('Test loss: {:.4f} - acc {:.4f}'.format(epoch_test_loss, epoch_test_acc))

    return epoch_test_loss, epoch_test_acc

def evaluate(dataset, parameters, steps_per_epoch, batch_size):
    sample_count = 0.
    cost = 0.
    accuracy = 0.

    for i in range(steps_per_epoch):
        X, Y = dataset.next_batch(batch_size)
        sample_count += int(X.shape[0])

        activations = forward_propagation(X, parameters, batch_size)
        cost += categorical_cross_entropy(activations['A3'], Y.T) * int(X.shape[0])
        accuracy += np.sum(np.argmax(activations['A3'].T, 1) == np.argmax(Y, 1))

    loss = cost / sample_count
    acc = accuracy / sample_count

    return loss, acc
