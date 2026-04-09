import unittest

import numpy as np

from scratchml import ElasticNetRegression, Lasso, LinearRegression, LogisticRegression, Ridge, SoftmaxRegression


class TestLinearModels(unittest.TestCase):
    def test_linear_regression_recovers_simple_line(self):
        X = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([3.0, 5.0, 7.0, 9.0])

        model = LinearRegression().fit(X, y)

        self.assertAlmostEqual(model.intercept_, 1.0, places=6)
        self.assertTrue(np.allclose(model.coef_, [2.0], atol=1e-6))
        self.assertAlmostEqual(model.predict(10.0), 21.0, places=6)

    def test_ridge_lasso_and_elastic_net_predict_scalars(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.1, 4.0, 5.9, 8.2])

        models = [
            Ridge(alpha=0.5),
            Lasso(alpha=0.01, lr=0.05, n_iters=3000),
            ElasticNetRegression(alpha=0.02, l1_ratio=0.4, lr=0.05, n_iters=3000),
        ]

        for model in models:
            model.fit(X, y)
            prediction = model.predict([5.0])
            self.assertIsInstance(prediction, float)
            self.assertGreater(prediction, 8.0)

    def test_logistic_regression_learns_binary_boundary(self):
        X = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
        y = np.array(["cold", "cold", "cold", "hot", "hot", "hot"])

        model = LogisticRegression(n_iters=4000, lr=0.2, random_state=0).fit(X, y)

        self.assertEqual(model.predict([-1.5]), "cold")
        self.assertEqual(model.predict([1.5]), "hot")
        probabilities = np.array(model.predict_proba([[0.0], [1.0]]))
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_softmax_regression_handles_multiclass(self):
        X = np.array(
            [
                [-2.0, -1.0],
                [-1.0, -1.5],
                [0.0, 2.0],
                [0.5, 1.8],
                [2.0, -1.0],
                [2.5, -0.5],
            ]
        )
        y = np.array([0, 0, 1, 1, 2, 2])

        model = SoftmaxRegression(n_iters=4000, lr=0.1, random_state=0).fit(X, y)

        preds = model.predict(X)
        self.assertEqual(preds, y.tolist())


if __name__ == "__main__":
    unittest.main()
