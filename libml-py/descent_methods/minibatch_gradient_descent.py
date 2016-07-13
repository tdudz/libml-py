"""
Minibatch Gradient Descent
---------------------------

Computes the gradient of the cost function with respect to the 
parameters for a set of n training examples.
"""

import numpy as np

def minibatch_gradient_descent(X, y, gradient, batch_size, learning_rate=0.01, iterations=1000):
    """
    Performs minibatch gradient descent for a set number of iterations.

    Args:
        X (np.ndarray): Training data with shape [n_samples, n_features]
        y (np.ndarray): Target values with shape [n_samples, 1]
        gradient (func): Function used to compute the gradient, a function of (X, y, weights)
        batch_size (int): Size of the individual minibatch
        learning rate (float): Step size used during each iteration
        iterations (int): Number of iterations to perform

    Returns:
        np.ndarray: A np array of weights with shape [n_features, 1]
    """
    theta = np.zeros((len(X[0]),1))
    for _ in xrange(iterations):
        shuffled = map(np.random.permutation, X)
        for i in xrange(0,len(y),batch_size):
            theta = np.subtract(theta, learning_rate * gradient(X[i:i+batch_size], y[i:i+batch_size], theta))
    return theta
