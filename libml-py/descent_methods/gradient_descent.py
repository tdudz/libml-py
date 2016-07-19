"""
Gradient Descent
---------------

Vanilla gradient descent. Supports batch, minibatch, and stochastic updating.

"""

import numpy as np 

def gradient_descent(X, y, gradient, theta=None, update="batch", alpha=0.01, iterations=1000):
	"""
	Performs batch gradient descent for a set number of iterations.

	Args:
		X (np.ndarray): training data with shape [n_samples, n_features]
		y (np.ndarray): target values with shape [n_samples, 1]
		gradient (func): function used to compute the gradient, a function of (X, y, weights)
		theta (np.ndarray): np array of weights/parameters to update
		update (str): method to use for updating weights, supports batch, minibatch, stochastic
		alpha (float): step size used during each iteration
		iterations (int): number of iterations to perform

	Returns:
		np.ndarray: a np array of weights with shape [n_features, 1]

	Raises:
		ValueError: if update method is unknown
	"""
	if theta is None:
		theta = np.zeros((len(X[0]),1))
		
	if update == "batch":
		for _ in xrange(iterations):
			theta = np.subtract(theta, learning_rate * gradient(X, y, theta))
		return theta

	elif update == "minibatch":
		for _ in xrange(iterations):
        	shuffled = map(np.random.permutation, X)
        	for i in xrange(0,len(y),batch_size):
        	    theta = np.subtract(theta, learning_rate * gradient(X[i:i+batch_size], y[i:i+batch_size], theta))
    	return theta

	elif update == "stochastic":
		for _ in xrange(iterations):
			locs = list(range(len(y)))
			np.random.shuffle(locs)
			for loc in locs:
				rand_x = np.asarray([X[loc]])
				rand_y = np.asarray([y[loc]])
				theta = np.subtract(theta, learning_rate * gradient(rand_x, rand_y, theta))
		return theta

	else:
		raise ValueError("Unknown update type specified.")
