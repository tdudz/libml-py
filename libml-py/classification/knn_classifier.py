"""
k-Nearest Neighbors Classification
----------------------------------
Supports L1 norm (Manhattan), L2 norm (Euclidean), Linf norm (Chebyshev).

"""

import numpy as numpy
import math

class kNearestNeighborsClassifier(object):

    def __init__(self):
        self.fitted = False
        self.dataset = None
        self.labels = None

    @staticmethod
    def euclidean_dist(p1, p2):
        """
        Given two points, calculates the euclidean distance (L2 norm) between them.

        Args:
            p1 (np.ndarray): the first point, of shape [n_features, 1]
            p2 (np.ndarray): the second point, of shape [n_features, 1]

        Returns:
            float: the euclidean distance between two points

        Raise:
            ValueError: if dimensions of inputs are not equal
        """
        if len(p1) != len(p2):
            raise ValueError("Dimensions of inputs are not equal.")
        return np.sum(np.square(p1 - p2))
        
    @staticmethod
    def manhattan_dist(p1, p2):
        """
        Given two points, calculates the manhattan distance (L1 norm) between them.

        Args:
            p1 (np.ndarray): the first point, of shape [n_features, 1]
            p2 (np.ndarray): the second point, of shape [n_features, 1]

        Returns:
            float: the euclidean distance between two points

        Raise:
            ValueError: if dimensions of inputs are not equal
        """
        if len(p1) != len(p2):
            raise ValueError("Dimensions of inputs are not equal.")
        return np.sum(np.fabs(p1 - p2))

    @staticmethod
    def chebyshev_dist(p1, p2):
        """
        Given two points, calculates the chebyshev distance (Linf norm) between them.

        Args:
            p1 (np.ndarray): the first point, of shape [n_features, 1]
            p2 (np.ndarray): the second point, of shape [n_features, 1]

        Returns:
            float: the euclidean distance between two points

        Raise:
            ValueError: if dimensions of inputs are not equal
        """

        if len(p1) != len(p2):
            raise ValueError("Dimensions of inputs are not equal.")
        return np.amax(np.fabs(p1 - p2))

    def fit(self, X, y, normalize_data=False):
        """
        Fits the model based on the training data.

        Args:
            X (np.ndarray): training data with shape [n_samples, n_features]
            y (np.ndarray): target values with shape [n_samples, 1]
            normalize_data (bool): whether or not to normalize input data
        """
        if normalize_data:
            mu = np.mean(X, axis=0)
            sigma = np.std(X, axis=0)
            X = (X-mu)/sigma
        self.dataset = X
        self.labels = y
        self.fitted = True

    def predict(self, X, k, dist="L2"):
        """
        Predicts an output for a given input vector based on the fitted model.
        Supports L1, L2, Linf distances.

        Args:
            X (np.ndarray): input data with shape [n_samples, n_features]
            k (int): number of neighbors to use
        
        Returns:
            np.ndarray: predicted output with shape [n_samples, 1]

        Raise:
            ValueError: if distance argument is unknown
        """
        if not self.fit:
            raise ValueError('Model must be fit before predicting.')
        for point in X:
            distances = []
            for i in xrange(len(self.dataset)):
                if dist == "L1":
                    distance = manhattan_dist(self.dataset[i], point)
                    distances.append(self.dataset[i], distance)
                elif dist == "L2":
                    distance = euclidean_dist(self.dataset[i], point)
                    distances.append(self.dataset[i], distance)
                elif dist == "Linf":
                    distance = chebyshev_dist(self.dataset[i], point)
                    distances.append(self.dataset[i], distance)
                else:
                    raise ValueError("Unknown distance type.")
            distances = sorted(distances, key=lambda x: x[1])
    
            neighbors = []
            for i in xrange(k):
                neighbors.append(distances[i][0])
