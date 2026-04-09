from __future__ import annotations

import numpy as np

from .utils import ensure_2d_features, ensure_targets, finalize_predictions, infer_single_input, validate_X_y


class _TreeNode:
    def __init__(self, *, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeRegressor:
    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2, max_features: int | None = None, random_state: int | None = None):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.tree_ = None
        self.n_features_in_ = None
        self._rng = None

    def fit(self, X, y):
        X, y = validate_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        predictions = np.array([self._predict_row(row, self.tree_) for row in X], dtype=float)
        return finalize_predictions(predictions, single_input=single_input)

    def _build_tree(self, X, y, depth: int):
        if self._should_stop(y, depth):
            return _TreeNode(value=float(np.mean(y)))

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return _TreeNode(value=float(np.mean(y)))

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return _TreeNode(feature_index=feature_index, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_loss = np.inf

        for feature_index in self._sample_features(X.shape[1]):
            values = np.unique(X[:, feature_index])
            if len(values) == 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                loss = self._weighted_mse(y[left_mask], y[right_mask])
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature_index
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _weighted_mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        total = len(left_y) + len(right_y)
        left_loss = np.mean((left_y - np.mean(left_y)) ** 2) if len(left_y) else 0.0
        right_loss = np.mean((right_y - np.mean(right_y)) ** 2) if len(right_y) else 0.0
        return (len(left_y) / total) * left_loss + (len(right_y) / total) * right_loss

    def _sample_features(self, n_features: int) -> np.ndarray:
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        return self._rng.choice(n_features, size=self.max_features, replace=False)

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        return (
            len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
        )

    def _predict_row(self, row: np.ndarray, node: _TreeNode):
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _check_is_fitted(self):
        if self.tree_ is None:
            raise RuntimeError("The model must be fit before calling predict().")


class DecisionTreeClassifier:
    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2, max_features: int | None = None, random_state: int | None = None):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.tree_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self._rng = None

    def fit(self, X, y):
        X, y = validate_X_y(X, y, classification=True)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        predictions = np.array([self._predict_row(row, self.tree_) for row in X], dtype=self.classes_.dtype)
        return finalize_predictions(predictions, single_input=single_input)

    def _build_tree(self, X, y, depth: int):
        if self._should_stop(y, depth):
            return _TreeNode(value=self._majority_class(y))

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return _TreeNode(value=self._majority_class(y))

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return _TreeNode(feature_index=feature_index, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        parent_impurity = self._gini(y)

        for feature_index in self._sample_features(X.shape[1]):
            values = np.unique(X[:, feature_index])
            if len(values) == 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = parent_impurity - self._weighted_gini(y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _gini(self, y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return float(1.0 - np.sum(probabilities ** 2))

    def _weighted_gini(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        total = len(left_y) + len(right_y)
        return (len(left_y) / total) * self._gini(left_y) + (len(right_y) / total) * self._gini(right_y)

    def _sample_features(self, n_features: int) -> np.ndarray:
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        return self._rng.choice(n_features, size=self.max_features, replace=False)

    def _majority_class(self, y: np.ndarray):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        return (
            len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
        )

    def _predict_row(self, row: np.ndarray, node: _TreeNode):
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _check_is_fitted(self):
        if self.tree_ is None or self.classes_ is None:
            raise RuntimeError("The model must be fit before calling predict().")
