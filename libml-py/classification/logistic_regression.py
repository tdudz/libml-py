"""
Logistic Regression
-----------------

Performs logistic regression, including regularization.

"""

import numpy as np

class LogisticRegression:

	def __init__(self):
		"""
		Attributes:
			fitted (bool): whether or not the model has been fit
			theta (np.ndarray): vector of weights with size [n_features, 1]
		"""
		self.fitted = False
		self.theta = np.NaN

	@staticmethod
	def sigmoid(X):
		"""
		Args:
			X (np.ndarray): array of inputs with shape [n_samples, 1]

		Returns:
			np.ndarray: output of sigmoid function with shape [n_samples, 1]
		"""
		return 1 / (1 + np.exp(-X))

	def gradient(self, X, y, theta):
		"""
		Computes the gradient of the cost function.

		Args:
			X (np.ndarray): training data with shape [n_samples, n_features]
			y (np.ndarray): target values with shape [n_samples, 1]
			theta (np.ndarray): an array of weights with shape [n_features, 1]

		Returns:
		 	np.ndarray: the gradient of the cost function
		"""
		m = len(y)
		gradient = X.T.dot(np.subtract(y, sigmoid(X.dot(theta))))
		return gradient

	def fit(self, X, y, normalize_data=False):
		"""
		Fits the model based on the training data.

		Args:
			X (np.ndarray): training data with shape [n_samples, n_features]
			y (np.ndarray): target values with shape [n_samples, 1]
			normalize_data (bool): whether or not to normalize input data
		"""
		X = np.insert(X, 0, 1, axis=1)
		self.theta = batch_gradient_descent(X, y, self.gradient)
		self.fitted = True

	def predict():
			"""
		Predicts an output for a given input vector based on the fitted model.

		Args:
			X (np.ndarray): input data with shape [n_samples, n_features]
		
		Returns:
			np.ndarray: predicted output with shape [n_samples, 1]

		Raise:
			ValueError: if model is not fit
		"""
		if not self.fit:
			raise ValueError('Model must be fit before predicting.')
		X = np.insert(X, 0, 1, axis=1)
		prediction = self.sigmoid(X.dot(self.theta))
		return np.round(prediction)



