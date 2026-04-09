from __future__ import annotations

import numpy as np

from .utils import ensure_2d_features


class StandardScaler:
    """Scale features to zero mean and unit variance."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = ensure_2d_features(X)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = ensure_2d_features(X)
        if X.shape[1] != len(self.mean_):
            raise ValueError("X has a different number of features than the fitted data.")
        return ((X - self.mean_) / self.scale_).tolist()

    def fit_transform(self, X):
        X = ensure_2d_features(X)
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        self._check_is_fitted()
        X = ensure_2d_features(X)
        if X.shape[1] != len(self.mean_):
            raise ValueError("X has a different number of features than the fitted data.")
        return (X * self.scale_ + self.mean_).tolist()

    def _check_is_fitted(self):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("The scaler must be fit before calling transform().")


class MinMaxScaler:
    """Scale features into a configurable range."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        min_value, max_value = feature_range
        if min_value >= max_value:
            raise ValueError("feature_range min must be less than max.")
        self.feature_range = (float(min_value), float(max_value))
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X):
        X = ensure_2d_features(X)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        data_range = self.data_max_ - self.data_min_
        self.data_range_ = np.where(data_range == 0.0, 1.0, data_range)
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = ensure_2d_features(X)
        if X.shape[1] != len(self.data_min_):
            raise ValueError("X has a different number of features than the fitted data.")
        low, high = self.feature_range
        normalized = (X - self.data_min_) / self.data_range_
        return (normalized * (high - low) + low).tolist()

    def fit_transform(self, X):
        X = ensure_2d_features(X)
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        self._check_is_fitted()
        X = ensure_2d_features(X)
        if X.shape[1] != len(self.data_min_):
            raise ValueError("X has a different number of features than the fitted data.")
        low, high = self.feature_range
        unscaled = (X - low) / (high - low)
        return (unscaled * self.data_range_ + self.data_min_).tolist()

    def _check_is_fitted(self):
        if self.data_min_ is None or self.data_max_ is None or self.data_range_ is None:
            raise RuntimeError("The scaler must be fit before calling transform().")
