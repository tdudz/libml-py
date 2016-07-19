"""
Utilities
---------

"""

import numpy as np

def normalize(X):
	"""
	Normalizes data to have mu = 0 and sigma = 1.

	Args:
		X (np.ndarray): input data of size [n_samples, n_features]

	Returns:
		np.ndarray: normalized version of input data of same size
	"""
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0)
	return (X - mu) / sigma
