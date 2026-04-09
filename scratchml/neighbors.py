from __future__ import annotations

import numpy as np

from .utils import ensure_2d_features, ensure_targets, finalize_predictions, infer_single_input, validate_X_y


class KNeighborsClassifier:
    def __init__(self, n_neighbors: int = 5):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = int(n_neighbors)
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        X, y = validate_X_y(X, y, classification=True)
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors cannot exceed the number of training samples.")
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        distances = np.linalg.norm(X[:, None, :] - self.X_train_[None, :, :], axis=2)
        neighbor_idx = np.argpartition(distances, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
        labels = self.y_train_[neighbor_idx]
        predictions = []
        for row in labels:
            values, counts = np.unique(row, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        return finalize_predictions(np.array(predictions), single_input=single_input)

    def _check_is_fitted(self):
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("The model must be fit before calling predict().")


class KNeighborsRegressor:
    def __init__(self, n_neighbors: int = 5):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = int(n_neighbors)
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        X, y = validate_X_y(X, y)
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors cannot exceed the number of training samples.")
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        distances = np.linalg.norm(X[:, None, :] - self.X_train_[None, :, :], axis=2)
        neighbor_idx = np.argpartition(distances, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
        predictions = np.mean(self.y_train_[neighbor_idx], axis=1)
        return finalize_predictions(predictions, single_input=single_input)

    def _check_is_fitted(self):
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("The model must be fit before calling predict().")
