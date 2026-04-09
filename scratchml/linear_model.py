from __future__ import annotations

import numpy as np

from .utils import (
    add_intercept,
    ensure_2d_features,
    finalize_predictions,
    infer_single_input,
    sigmoid,
    softmax,
    validate_X_y,
)


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X, y = validate_X_y(X, y)
        X_aug = add_intercept(X)
        weights, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        preds = X @ self.coef_ + self.intercept_
        return finalize_predictions(preds, single_input=single_input)

    def _check_is_fitted(self):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model must be fit before calling predict().")


class Ridge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X, y = validate_X_y(X, y)
        X_aug = add_intercept(X)
        penalty = np.eye(X_aug.shape[1])
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(X_aug.T @ X_aug + self.alpha * penalty, X_aug.T @ y)
        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        preds = X @ self.coef_ + self.intercept_
        return finalize_predictions(preds, single_input=single_input)

    def _check_is_fitted(self):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model must be fit before calling predict().")


class Lasso:
    def __init__(self, alpha: float = 0.1, lr: float = 1e-2, n_iters: int = 5000, tol: float = 1e-6):
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.tol = float(tol)
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

    def fit(self, X, y):
        X, y = validate_X_y(X, y)
        X_aug = add_intercept(X)
        weights = np.zeros(X_aug.shape[1])
        previous_objective = np.inf

        for _ in range(self.n_iters):
            residual = X_aug @ weights - y
            gradient = (2.0 / len(X_aug)) * (X_aug.T @ residual)
            candidate = weights - self.lr * gradient
            candidate[1:] = self._soft_threshold(candidate[1:], self.lr * self.alpha)
            weights = candidate

            mse = float(np.mean((X_aug @ weights - y) ** 2))
            objective = mse + self.alpha * float(np.sum(np.abs(weights[1:])))
            if abs(previous_objective - objective) < self.tol:
                break
            previous_objective = objective

        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        preds = X @ self.coef_ + self.intercept_
        return finalize_predictions(preds, single_input=single_input)

    def _check_is_fitted(self):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model must be fit before calling predict().")


class ElasticNetRegression:
    def __init__(
        self,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        lr: float = 1e-2,
        n_iters: int = 5000,
        tol: float = 1e-6,
    ):
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be between 0 and 1.")
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.tol = float(tol)
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

    def fit(self, X, y):
        X, y = validate_X_y(X, y)
        X_aug = add_intercept(X)
        weights = np.zeros(X_aug.shape[1])
        previous_objective = np.inf
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)

        for _ in range(self.n_iters):
            residual = X_aug @ weights - y
            gradient = (2.0 / len(X_aug)) * (X_aug.T @ residual)
            candidate = weights - self.lr * gradient
            candidate[1:] = self._soft_threshold(candidate[1:], self.lr * l1_penalty)
            candidate[1:] /= 1.0 + 2.0 * self.lr * l2_penalty
            weights = candidate

            mse = float(np.mean((X_aug @ weights - y) ** 2))
            objective = mse + self.alpha * (
                self.l1_ratio * float(np.sum(np.abs(weights[1:])))
                + (1.0 - self.l1_ratio) * float(np.sum(weights[1:] ** 2))
            )
            if abs(previous_objective - objective) < self.tol:
                break
            previous_objective = objective

        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        preds = X @ self.coef_ + self.intercept_
        return finalize_predictions(preds, single_input=single_input)

    def _check_is_fitted(self):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model must be fit before calling predict().")


class LogisticRegression:
    def __init__(self, n_iters: int = 1000, lr: float = 0.1, tol: float = 1e-6, random_state: int | None = None):
        self.n_iters = int(n_iters)
        self.lr = float(lr)
        self.tol = float(tol)
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X, y):
        X, y = validate_X_y(X, y, classification=True)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("LogisticRegression expects exactly two classes.")

        self.classes_ = classes
        y_binary = (y == classes[1]).astype(float)
        X_aug = add_intercept(X)
        rng = np.random.default_rng(self.random_state)
        weights = rng.normal(loc=0.0, scale=0.01, size=X_aug.shape[1])

        for _ in range(self.n_iters):
            probabilities = sigmoid(X_aug @ weights)
            gradient = X_aug.T @ (probabilities - y_binary) / len(X_aug)
            updated = weights - self.lr * gradient
            if np.linalg.norm(updated - weights) < self.tol:
                weights = updated
                break
            weights = updated

        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict_proba(self, X):
        self._check_is_fitted()
        X = ensure_2d_features(X)
        positive = sigmoid(X @ self.coef_ + self.intercept_)
        negative = 1.0 - positive
        return np.column_stack([negative, positive]).tolist()

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        positive = sigmoid(X @ self.coef_ + self.intercept_)
        predictions = np.where(positive >= 0.5, self.classes_[1], self.classes_[0])
        return finalize_predictions(predictions, single_input=single_input)

    def _check_is_fitted(self):
        if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
            raise RuntimeError("The model must be fit before calling predict().")


class SoftmaxRegression:
    def __init__(self, lr: float = 0.05, n_iters: int = 2000, tol: float = 1e-6, random_state: int | None = None):
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.tol = float(tol)
        self.random_state = random_state
        self.weights_ = None
        self.classes_ = None

    def fit(self, X, y):
        X, y = validate_X_y(X, y, classification=True)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        X_aug = add_intercept(X)
        rng = np.random.default_rng(self.random_state)
        weights = rng.normal(loc=0.0, scale=0.01, size=(X_aug.shape[1], len(self.classes_)))
        y_one_hot = np.zeros((len(y_encoded), len(self.classes_)))
        y_one_hot[np.arange(len(y_encoded)), y_encoded] = 1.0

        for _ in range(self.n_iters):
            probabilities = softmax(X_aug @ weights)
            gradient = X_aug.T @ (probabilities - y_one_hot) / len(X_aug)
            updated = weights - self.lr * gradient
            if np.linalg.norm(updated - weights) < self.tol:
                weights = updated
                break
            weights = updated

        self.weights_ = weights
        return self

    def predict_proba(self, X):
        self._check_is_fitted()
        X = ensure_2d_features(X)
        probabilities = softmax(add_intercept(X) @ self.weights_)
        return probabilities.tolist()

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        probabilities = softmax(add_intercept(X) @ self.weights_)
        labels = self.classes_[np.argmax(probabilities, axis=1)]
        return finalize_predictions(labels, single_input=single_input)

    def _check_is_fitted(self):
        if self.weights_ is None or self.classes_ is None:
            raise RuntimeError("The model must be fit before calling predict().")
