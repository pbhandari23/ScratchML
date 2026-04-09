"""Run a small benchmark for the scratch ML implementations."""

from __future__ import annotations

import numpy as np

from scratchml import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    KMeans,
    KNeighborsClassifier,
    KNeighborsRegressor,
    LinearRegression,
    LogisticRegression,
    RandomForestClassifier,
    RandomForestRegressor,
    SoftmaxRegression,
    accuracy_score,
    mean_squared_error,
    train_test_split,
)


def make_regression_data(n_samples=300, noise=0.15, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 3))
    weights = np.array([2.5, -1.2, 0.8])
    y = X @ weights + 1.5 + rng.normal(scale=noise, size=n_samples)
    return X, y


def make_binary_classification_data(n_samples=300, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 2))
    boundary = 1.8 * X[:, 0] - 1.1 * X[:, 1] + 0.2
    y = (boundary > 0).astype(int)
    return X, y


def make_multiclass_data(n_samples=360, random_state=42):
    rng = np.random.default_rng(random_state)
    centers = np.array([[-3.0, -1.0], [0.0, 3.0], [3.0, -1.0]])
    X_parts = []
    y_parts = []
    for label, center in enumerate(centers):
        X_parts.append(rng.normal(loc=center, scale=0.8, size=(n_samples // 3, 2)))
        y_parts.append(np.full(n_samples // 3, label))
    return np.vstack(X_parts), np.concatenate(y_parts)


def adjusted_rand_index(labels_true, labels_pred) -> float:
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    if n != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have the same length.")

    classes_true, inverse_true = np.unique(labels_true, return_inverse=True)
    classes_pred, inverse_pred = np.unique(labels_pred, return_inverse=True)
    contingency = np.zeros((len(classes_true), len(classes_pred)), dtype=int)
    for i in range(n):
        contingency[inverse_true[i], inverse_pred[i]] += 1

    def comb2(values):
        values = np.asarray(values, dtype=float)
        return np.sum(values * (values - 1) / 2.0)

    sum_comb_c = comb2(contingency)
    sum_comb_rows = comb2(contingency.sum(axis=1))
    sum_comb_cols = comb2(contingency.sum(axis=0))
    total_pairs = n * (n - 1) / 2.0
    expected_index = (sum_comb_rows * sum_comb_cols) / total_pairs if total_pairs else 0.0
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0
    return float((sum_comb_c - expected_index) / denominator)


def compare_regression():
    X, y = make_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    models = [
        ("LinearRegression", LinearRegression()),
        ("KNeighborsRegressor", KNeighborsRegressor(n_neighbors=5)),
        ("DecisionTreeRegressor", DecisionTreeRegressor(max_depth=4, random_state=42)),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=20, max_depth=4, max_features=2, random_state=42)),
    ]

    print("\nREGRESSION")
    print("-" * 60)
    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"{name:<24} MSE: {mean_squared_error(y_test, predictions):.4f}")


def compare_classification():
    X, y = make_binary_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    models = [
        ("LogisticRegression", LogisticRegression(n_iters=2000, lr=0.1, random_state=42)),
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=25, max_depth=4, max_features=1, random_state=42)),
    ]

    print("\nBINARY CLASSIFICATION")
    print("-" * 60)
    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"{name:<24} Accuracy: {accuracy_score(y_test, predictions):.4f}")


def compare_multiclass():
    X, y = make_multiclass_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = SoftmaxRegression(n_iters=2500, lr=0.05, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nMULTICLASS CLASSIFICATION")
    print("-" * 60)
    print(f"{'SoftmaxRegression':<24} Accuracy: {accuracy_score(y_test, predictions):.4f}")


def compare_clustering():
    X, y = make_multiclass_data()
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)
    predictions = model.predict(X)
    print("\nCLUSTERING")
    print("-" * 60)
    print(f"{'KMeans':<24} Inertia: {model.inertia_:.2f}")
    print(f"{'KMeans':<24} Adjusted Rand Index: {adjusted_rand_index(y, predictions):.4f}")


if __name__ == "__main__":
    compare_regression()
    compare_classification()
    compare_multiclass()
    compare_clustering()
