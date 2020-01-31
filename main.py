import os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(8)

from train import *

def main(args):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    test_cost, test_accuracy = train(mnist,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        alpha=args.alpha)

    return test_cost, test_accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--alpha', type=int, default=0.01)

    score = main(parser.parse_known_args()[0])
