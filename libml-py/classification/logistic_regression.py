"""
Logistic Regression
-----------------

Performs logistic regression, including regularization.
Includes closed form and descent method solving.

"""

import numpy as np

class LinearRegression:
	"""Class for a logistic regression classifier."""
	def __init__(self):
		"""
		Attributes:
			fitted (bool): Whether or not the model has been fit
			theta (np.ndarray): vector of weights with size [n_features, 1]
		"""
		self.fitted = False
		self.theta = np.NaN

	def fit(self, X, y, normalize_data=False, descent=True):

	def predict(self, X):
