from __future__ import annotations

import numpy as np

from .tree import DecisionTreeClassifier, DecisionTreeRegressor
from .utils import ensure_2d_features, finalize_predictions, infer_single_input, validate_X_y


class RandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | None = None,
        random_state: int | None = None,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, y):
        X, y = validate_X_y(X, y)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []

        for _ in range(self.n_estimators):
            sample_idx = rng.choice(len(X), size=len(X), replace=True)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees_], dtype=float)
        predictions = np.mean(tree_predictions, axis=0)
        return finalize_predictions(predictions, single_input=single_input)

    def _check_is_fitted(self):
        if not self.trees_:
            raise RuntimeError("The model must be fit before calling predict().")


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | None = None,
        random_state: int | None = None,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, y):
        X, y = validate_X_y(X, y, classification=True)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []

        for _ in range(self.n_estimators):
            sample_idx = rng.choice(len(X), size=len(X), replace=True)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees_])
        predictions = []
        for column in tree_predictions.T:
            values, counts = np.unique(column, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        return finalize_predictions(np.array(predictions), single_input=single_input)

    def _check_is_fitted(self):
        if not self.trees_:
            raise RuntimeError("The model must be fit before calling predict().")
