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
		X (np.ndarray): Training data with shape [n_samples, n_features]
		y (np.ndarray): Target values with shape [n_samples, 1]
		gradient (func): Function used to compute the gradient, a function of (X, y, weights)
		learning rate (float): Step size used during each iteration
		iterations (int): Number of iterations to perform

	Returns:
		np.ndarray: A np array of weights with shape [n_features, 1]
	"""
	theta = np.zeros((len(X[0]),1))
	for _ in xrange(iterations):
		locs = list(range(len(y)))
		np.random.shuffle(locs)
		for loc in locs:
			rand_x = np.asarray([X[loc]])
			rand_y = np.asarray([y[loc]])
			theta = np.subtract(theta, learning_rate * gradient(rand_x, rand_y, theta))
	return theta
