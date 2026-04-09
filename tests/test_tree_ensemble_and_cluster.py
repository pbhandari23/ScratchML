import unittest

import numpy as np

from scratchml import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    KMeans,
    KNeighborsClassifier,
    KNeighborsRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    accuracy_score,
    mean_squared_error,
)


class TestTreeEnsembleAndClusterModels(unittest.TestCase):
    def test_knn_models_work_on_small_examples(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_class = np.array([0, 0, 1, 1])
        y_reg = np.array([0.0, 1.0, 2.0, 3.0])

        clf = KNeighborsClassifier(n_neighbors=3).fit(X, y_class)
        reg = KNeighborsRegressor(n_neighbors=2).fit(X, y_reg)

        self.assertEqual(clf.predict([2.5]), 1)
        self.assertAlmostEqual(reg.predict([2.5]), 2.5, places=6)

    def test_decision_tree_models_fit_clean_signals(self):
        X_reg = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y_reg = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        tree_reg = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_reg, y_reg)
        self.assertLess(mean_squared_error(y_reg, tree_reg.predict(X_reg)), 1e-9)

        X_clf = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y_clf = np.array([0, 0, 0, 1, 1])
        tree_clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_clf, y_clf)
        self.assertEqual(tree_clf.predict([3.5]), 1)

    def test_random_forests_produce_strong_training_scores(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(80, 3))
        y_reg = 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(scale=0.1, size=80)
        y_clf = (X[:, 0] + X[:, 1] > 0).astype(int)

        forest_reg = RandomForestRegressor(n_estimators=20, max_depth=5, max_features=2, random_state=7).fit(X, y_reg)
        forest_clf = RandomForestClassifier(n_estimators=20, max_depth=5, max_features=2, random_state=7).fit(X, y_clf)

        self.assertLess(mean_squared_error(y_reg, forest_reg.predict(X)), 0.25)
        self.assertGreater(accuracy_score(y_clf, forest_clf.predict(X)), 0.9)

    def test_kmeans_finds_cluster_assignments(self):
        rng = np.random.default_rng(4)
        cluster_a = rng.normal(loc=[-3.0, 0.0], scale=0.3, size=(20, 2))
        cluster_b = rng.normal(loc=[3.0, 0.0], scale=0.3, size=(20, 2))
        X = np.vstack([cluster_a, cluster_b])

        model = KMeans(n_clusters=2, random_state=4).fit(X)
        preds = np.asarray(model.predict(X))

        self.assertEqual(preds.shape, (40,))
        self.assertEqual(len(np.unique(preds)), 2)
        self.assertGreater(model.inertia_, 0.0)


if __name__ == "__main__":
    unittest.main()
