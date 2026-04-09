from __future__ import annotations

import numpy as np

from .utils import ensure_2d_features, finalize_predictions, infer_single_input


class KMeans:
    def __init__(self, n_clusters: int, n_iters: int = 100, tol: float = 1e-4, random_state: int | None = None):
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        self.n_clusters = int(n_clusters)
        self.n_iters = int(n_iters)
        self.tol = float(tol)
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        X = ensure_2d_features(X)
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed the number of samples.")

        rng = np.random.default_rng(self.random_state)
        centroids = X[rng.choice(len(X), size=self.n_clusters, replace=False)]

        for _ in range(self.n_iters):
            distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            updated = centroids.copy()
            for cluster_id in range(self.n_clusters):
                members = X[labels == cluster_id]
                if len(members) == 0:
                    updated[cluster_id] = X[rng.integers(0, len(X))]
                else:
                    updated[cluster_id] = members.mean(axis=0)

            shift = float(np.linalg.norm(updated - centroids))
            centroids = updated
            if shift <= self.tol:
                break

        self.centroids_ = centroids
        final_distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        self.labels_ = np.argmin(final_distances, axis=1)
        self.inertia_ = float(np.sum((X - centroids[self.labels_]) ** 2))
        return self

    def predict(self, X):
        self._check_is_fitted()
        single_input = infer_single_input(X)
        X = ensure_2d_features(X)
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        return finalize_predictions(labels, single_input=single_input)

    def _check_is_fitted(self):
        if self.centroids_ is None:
            raise RuntimeError("The model must be fit before calling predict().")
