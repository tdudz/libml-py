"""
Newton's Method
---------------

"""

import numpy as np

def newtons_method(X, y, gradient, hessian, learning_rate=0.01, iterations=10000):
	"""
	Performs newton's method for a set number of iterations, with a constant step size.

	Args:
		X (np.ndarray): training data with shape [n_samples, n_features]
		y (np.ndarray): target values with shape [n_samples, 1]
		gradient (func): function used to compute the gradient, is a function of (X, y, theta)
		hessian (func): function used to compute the hessian, is a function of (X)
		learning rate (float): step size used during each iteration
		iterations (int): number of iterations to perform

	Returns:
		np.ndarray: a np array of weights with shape [n_features, 1]
	"""
	theta = np.zeros((len(X[0]),1))
	for _ in xrange(iterations):
		newton_step = np.linalg.pinv(hessian(X)).dot(gradient(X, y, theta))
		theta = np.subtract(theta, learning_rate * newton_step)
	return theta
