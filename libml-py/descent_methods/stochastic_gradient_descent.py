"""
Stochastic Gradient Descent
---------------------------

Computes the gradient of the cost function with respect to the 
parameters for each training example.

"""

import numpy as np

def stochastic_gradient_descent(X, y, gradient, learning_rate=0.01, iterations=1000):
	"""
	Performs stochastic gradient descent for a set number of iterations.

	Args:
		X (np.ndarray): training data with shape [n_samples, n_features]
		y (np.ndarray): target values with shape [n_samples, 1]
		gradient (func): function used to compute the gradient, a function of (X, y, weights)
		theta (np.ndarray): np array of weights/parameters to update
		learning rate (float): step size used during each iteration
		iterations (int): number of iterations to perform

	Returns:
		np.ndarray: a np array of weights with shape [n_features, 1]
	"""
	if theta is None:
		theta = np.zeros((len(X[0]),1))
	for _ in xrange(iterations):
		locs = list(range(len(y)))
		np.random.shuffle(locs)
		for loc in locs:
			rand_x = np.asarray([X[loc]])
			rand_y = np.asarray([y[loc]])
			theta = np.subtract(theta, learning_rate * gradient(rand_x, rand_y, theta))
	return theta
