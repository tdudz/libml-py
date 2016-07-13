"""
Linear Regression
-----------------

Performs linear regression, includes descent and closed form methods.

"""

import numpy as np
from batch_gradient_descent import batch_gradient_descent

class LinearRegression:
	"""Class for performing linear regression."""
	def __init__(self):
		"""
		Attributes:
			fitted (bool): Whether or not the model has been fit
			theta (np.ndarray): vector of weights with size [n_features, 1]
		"""
		self.fitted = False
		self.theta = np.NaN

	def gradient(self, X, y, theta):
		"""
		Computes the gradient of the cost function (MSE).
		J(theta) = 1/(2*m) * (y - X*theta)^2
		grad(J(Theta)) = 1/m * (y - X*theta) * X

		Args:
			X (np.ndarray): Training data with shape [n_samples, n_features]
			y (np.ndarray): Target values with shape [n_samples, 1]
			theta (np.ndarray): An array of weights with shape [n_features, 1]

		Returns:
		 	np.ndarray: The gradient of the MSE cost function
		"""
		m = len(y)
		gradient = 1./m * X.T.dot(np.subtract(X.dot(theta), y))
		return gradient

	def fit(self, X, y, normalize_data=False, descent=True):
		"""
		Fits the model based on the training data.

		Args:
			X (np.ndarray): Training data with shape [n_samples, n_features]
			y (np.ndarray): Target values with shape [n_samples, 1]
			normalize_data (bool): Whether or not to normalize input data
			descent(bool): Whether to solve using a descent or normal equations method
		"""
		if normalize_data:
			mu = np.mean(X, axis=0)
			sigma = np.std(X, axis=0)
			X = (X-mu)/sigma
		X = np.insert(X, 0, 1, axis=1)
		if descent:
			self.theta = batch_gradient_descent(X, y, self.gradient)
		else:
			self.theta = np.dot(np.linalg.pinv(X.T.dot(X)), X.T.dot(y))
		self.fitted = True

	def predict(self, X):
		"""
		Predicts an output for a given input vector based on the fitted model.

		Args:
			X (np.ndarray): Input data with shape [n_samples, n_features]
		
		Returns:
			np.ndarray: Predicted output with shape [n_samples, 1]

		Raise:
			ValueError: If model is not fit.
		"""
		if not self.fit:
			raise ValueError('Model must be fit before predicting.')
		X = np.insert(X, 0, 1, axis=1)
		prediction = X.dot(self.theta)
		return prediction
