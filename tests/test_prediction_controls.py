"""Reject invalid post-fit block sizes without losing a usable fitted model."""
import pickle
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from two_stage_sgd_dwd import KernelSGDDWD, TwoStageDWDClassifier


def linear_kernel(X, Y):
    return X @ Y.T


class PredictionControlTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[.1, .2], [.2, .3], [1., .8], [1.2, 1.]])
        self.y = np.array([-1, -1, 1, 1])
        self.query = np.vstack((self.X, self.X[:3] * .9))
        self.query_labels = np.r_[self.y, self.y[:3]]

    def models(self):
        for kernel, backend, gamma in (
            ("rbf", "direct", .25), ("rbf", "blas", .25),
            ("rbf", "blas", 0.), ("linear", "blas", .25),
            ("laplacian", "blas", .25), ("cosine", "blas", .25),
            ("poly", "blas", .25), ("chi2", "blas", .25),
            (linear_kernel, "blas", .25),
        ):
            yield KernelSGDDWD(kernel=kernel, prediction_backend=backend,
                gamma=gamma, max_epochs=3, random_state=11).fit(self.X, self.y)

    def test_invalid_post_fit_sizes_reject_all_public_readouts(self):
        for model in self.models():
            for size in (-256, -1, 0, True, False, np.bool_(True),
                         1.5, np.nan, np.inf, "2", None):
                model.set_params(prediction_batch_size=size)
                operations = (
                    lambda: model.decision_function(self.query),
                    lambda: model.predict(self.query),
                    lambda: model.objective(self.query, self.query_labels),
                    model.rkhs_norm_squared,
                )
                for index, operation in enumerate(operations):
                    with self.subTest(kernel=model.kernel, backend=model.prediction_backend,
                                      size=size, operation=index):
                        with self.assertRaisesRegex(ValueError, "prediction_batch_size"):
                            operation()

    def test_valid_sizes_and_recovery_preserve_outputs_and_model(self):
        for model in self.models():
            for size in (1, 3, np.int64(4), 100):
                with self.subTest(kernel=model.kernel, backend=model.prediction_backend,
                                  size=size):
                    model.set_params(prediction_batch_size=size)
                    state = pickle.dumps(model)
                    scores = model.decision_function(self.query)
                    labels = model.predict(self.query)
                    norm = model.rkhs_norm_squared()
                    objective = model.objective(self.query, self.query_labels)
                    model.set_params(prediction_batch_size=-1)
                    with self.assertRaises(ValueError):
                        model.predict(self.query)
                    with self.assertRaises(ValueError):
                        model.rkhs_norm_squared()
                    model.set_params(prediction_batch_size=size)
                    self.assertEqual(model.decision_function(self.query).tobytes(), scores.tobytes())
                    np.testing.assert_array_equal(model.predict(self.query), labels)
                    self.assertEqual(model.rkhs_norm_squared(), norm)
                    self.assertEqual(model.objective(self.query, self.query_labels), objective)
                    self.assertEqual(pickle.dumps(model), state)

    def test_two_stage_prediction_uses_child_control_validation(self):
        model = TwoStageDWDClassifier(retention_fraction=1., gamma=.25,
            max_epochs=3, random_state=11).fit(self.X, self.y)
        expected = model.decision_function(self.query)
        child = model.kernel_estimator_
        size = child.prediction_batch_size
        child.set_params(prediction_batch_size=-1)
        with self.assertRaisesRegex(ValueError, "prediction_batch_size"):
            model.predict(self.query)
        child.set_params(prediction_batch_size=size)
        self.assertEqual(model.decision_function(self.query).tobytes(), expected.tobytes())

    def test_unfitted_error_precedes_block_size_validation(self):
        model = KernelSGDDWD(prediction_batch_size=-1)
        for operation in (lambda: model.decision_function(self.query),
                          model.rkhs_norm_squared):
            with self.subTest(operation=operation):
                with self.assertRaises(NotFittedError):
                    operation()


if __name__ == "__main__":
    unittest.main()
