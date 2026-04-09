from __future__ import annotations

import numpy as np


def ensure_2d_features(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 0:
        return X.reshape(1, 1)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a scalar, 1D array, or 2D array.")
    return X


def ensure_targets(y, *, classification: bool = False) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 0:
        y = y.reshape(1)
    if y.ndim > 2:
        raise ValueError("y must be a scalar, 1D array, or column vector.")
    y = y.ravel()
    return y if classification else y.astype(float)


def validate_X_y(X, y, *, classification: bool = False) -> tuple[np.ndarray, np.ndarray]:
    X = ensure_2d_features(X)
    y = ensure_targets(y, classification=classification)
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if len(X) == 0:
        raise ValueError("X and y must contain at least one sample.")
    return X, y


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def finalize_predictions(values: np.ndarray | list[float] | list[int], *, single_input: bool):
    arr = np.asarray(values)
    if single_input:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def infer_single_input(X) -> bool:
    arr = np.asarray(X)
    return arr.ndim in (0, 1)


def mean_squared_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_test_split(X, y, test_size: float = 0.2, random_state: int | None = None):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X = ensure_2d_features(X)
    y = ensure_targets(y, classification=True)
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if len(X) == 0:
        raise ValueError("X and y must contain at least one sample.")
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
