import unittest

import numpy as np

from scratchml import MinMaxScaler, StandardScaler


class TestPreprocessing(unittest.TestCase):
    def test_standard_scaler_fit_transform_and_inverse(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        scaler = StandardScaler()

        X_scaled = np.array(scaler.fit_transform(X))

        self.assertTrue(np.allclose(X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-9))
        self.assertTrue(np.allclose(X_scaled.std(axis=0), [1.0, 1.0], atol=1e-9))
        self.assertTrue(np.allclose(scaler.inverse_transform(X_scaled), X, atol=1e-9))

    def test_standard_scaler_handles_constant_column(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaler = StandardScaler().fit(X)

        X_scaled = np.array(scaler.transform(X))

        self.assertTrue(np.allclose(X_scaled[:, 0], 0.0))

    def test_minmax_scaler_fit_transform_and_inverse(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))

        X_scaled = np.array(scaler.fit_transform(X))

        self.assertTrue(np.allclose(X_scaled.min(axis=0), [-1.0, -1.0], atol=1e-9))
        self.assertTrue(np.allclose(X_scaled.max(axis=0), [1.0, 1.0], atol=1e-9))
        self.assertTrue(np.allclose(scaler.inverse_transform(X_scaled), X, atol=1e-9))

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().transform([[1.0], [2.0]])


if __name__ == "__main__":
    unittest.main()
