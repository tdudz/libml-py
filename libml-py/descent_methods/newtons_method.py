"""
Newton's Method
---------------


"""

import numpy as np

def newtons_method(X, y, gradient, hessian, learning_rate=0.01, iterations=10000):
	"""
	Performs newton's method for a set number of iterations, with a constant step size.

	Args:
		X (np.ndarray): Training data with shape [n_samples, n_features]
		y (np.ndarray): Target values with shape [n_samples, 1]
		gradient (func): Function used to compute the gradient, is a function of (X, y, weights)
		hessian (func): Function used to compute the hessian, is a function of (X)
		learning rate (float): Step size used during each iteration
		iterations (int): Number of iterations to perform

	Returns:
		np.ndarray: A np array of weights with shape [n_features, 1]
	"""