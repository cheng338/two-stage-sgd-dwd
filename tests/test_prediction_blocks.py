"""Prediction parity and bounded-allocation regressions on synthetic data."""
import copy
import pickle
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist

from two_stage_sgd_dwd import KernelSGDDWD, TwoStageDWDClassifier
from two_stage_sgd_dwd import _prediction
from two_stage_sgd_dwd._estimators import _scores


def linear_kernel(X, Y):
    return X @ Y.T


class PredictionBlockTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(213)
        self.X = rng.normal(size=(12, 3))
        self.y = np.tile([-1, 1], 6)
        self.query = rng.normal(size=(11, 3))
        self.model = KernelSGDDWD(gamma=0.4, max_epochs=2, random_state=8,
                                 prediction_backend="direct").fit(self.X, self.y)

    @staticmethod
    def original(model, X):
        return _scores("kernel", np.asarray(X, dtype=np.float64),
                       model.dual_coef_[0], model.intercept_[0], model.basis_,
                       model.gamma_, model.prediction_batch_size,
                       kernel_spec=model._kernel_spec_)

    def assert_bits(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_exact_scores_across_blocks_and_both_estimators(self):
        two_stage = TwoStageDWDClassifier(retention_fraction=1.0, gamma=0.4,
                                          max_epochs=2, random_state=8,
                                          prediction_backend="direct").fit(self.X, self.y)
        for model in (self.model, two_stage):
            kernel = getattr(model, "kernel_estimator_", model)
            for size in (1, 2, 4, 11, 256):
                with self.subTest(model=type(model).__name__, batch_size=size):
                    kernel.prediction_batch_size = size
                    expected = self.original(kernel, self.query)
                    self.assert_bits(model.decision_function(self.query), expected)
                    np.testing.assert_array_equal(model.predict(self.query),
                                                  model.classes_[(expected >= 0).astype(int)])

    def test_exact_noncontiguous_fortran_readonly_and_high_offset_inputs(self):
        # Subtracting two large, close coordinates must still use direct cdist;
        # a norm/dot-product distance identity would suffer cancellation here.
        for offset in (0.0, 1e12):
            basis = self.X + offset
            query = self.query + offset
            layouts = [lambda x: x.copy(), np.asfortranarray,
                       lambda x: np.repeat(x, 2, axis=1)[:, ::2],
                       lambda x: x[::-1]]
            for basis_layout in layouts:
                for query_layout in layouts:
                    with self.subTest(offset=offset, basis=basis_layout, query=query_layout):
                        self.model.basis_ = basis_layout(basis)
                        self.model.basis_.flags.writeable = False
                        values = query_layout(query)
                        values.flags.writeable = False
                        self.model.prediction_batch_size = 4
                        before_basis = self.model.basis_.copy()
                        before_query = values.copy()
                        self.assert_bits(self.model.decision_function(values), self.original(self.model, values))
                        np.testing.assert_array_equal(self.model.basis_, before_basis)
                        np.testing.assert_array_equal(values, before_query)
        self.assert_bits(self.model.decision_function(self.query.astype(np.float32)),
                         self.original(self.model, self.query.astype(np.float32)))

    def test_one_bounded_cdist_output_is_reused_including_short_final_block(self):
        self.model.prediction_batch_size = 4
        calls = []

        def evaluate(left, right, *, metric, out):
            calls.append((left.shape, right.shape, metric, out, out.ctypes.data))
            self.assertEqual(out.dtype, np.float64)
            self.assertTrue(out.flags.c_contiguous)
            return cdist(left, right, metric=metric, out=out)

        with patch.object(_prediction, "cdist", side_effect=evaluate):
            actual = self.model.decision_function(self.query)
        self.assert_bits(actual, self.original(self.model, self.query))
        self.assertEqual([row[0][0] for row in calls], [4, 4, 3])
        self.assertEqual({row[2] for row in calls}, {"sqeuclidean"})
        self.assertEqual(len({row[4] for row in calls}), 1)
        self.assertEqual(calls[0][3].base.nbytes, 4 * len(self.X) * 8)
        self.assertTrue(all(row[3].base is calls[0][3].base for row in calls))

    def test_basis_and_gamma_are_checked_once_per_prediction_call(self):
        self.model.prediction_batch_size = 1
        with patch.object(_prediction, "_finite_array", wraps=_prediction._finite_array) as arrays, \
             patch.object(_prediction, "_finite_scalar", wraps=_prediction._finite_scalar) as scalars:
            self.model.decision_function(self.query)
        self.assertEqual(arrays.call_count, 1)
        self.assertEqual(arrays.call_args.args[1], "Y")
        self.assertEqual(scalars.call_count, 1)

    def test_invalid_public_queries_still_raise(self):
        bad_inputs = [np.full((2, 3), np.nan), np.full((2, 3), np.inf),
                      np.ones((2, 3), dtype=complex) * (1 + 1j), np.ones((2, 2)),
                      np.empty((0, 3)), np.ones(3)]
        for values in bad_inputs:
            with self.subTest(shape=values.shape, dtype=values.dtype):
                with self.assertRaises(ValueError):
                    self.model.decision_function(values)

    def test_mutated_basis_validation_matches_original(self):
        bad_bases = [np.full((12, 3), np.nan), np.full((12, 3), np.inf),
                     self.X.astype(complex) + 1j, np.ones((12, 2)), np.ones(12)]
        for basis in bad_bases:
            with self.subTest(shape=basis.shape):
                self.model.basis_ = basis
                with self.assertRaises(ValueError):
                    self.original(self.model, self.query)
                with self.assertRaises(ValueError):
                    self.model.decision_function(self.query)

    def test_mutated_gamma_preserves_original_routing(self):
        for gamma in (0.0, -1.0, np.nan):
            with self.subTest(gamma=gamma):
                self.model.gamma_ = gamma
                with patch("two_stage_sgd_dwd._estimators.rbf_prediction_scores",
                           side_effect=AssertionError("Nonpositive/NaN route changed")):
                    self.assert_bits(self.model.decision_function(self.query),
                                     self.original(self.model, self.query))
        self.model.gamma_ = np.inf
        with self.assertRaises(ValueError):
            self.original(self.model, self.query)
        with self.assertRaises(ValueError):
            self.model.decision_function(self.query)

    def test_distance_and_width_overflow_match_original_failures(self):
        cases = [(np.full((12, 3), 1e308), np.full((2, 3), -1e308), 0.4),
                 (np.zeros((12, 3)), np.full((2, 3), 2.0), 1e308)]
        for basis, query, gamma in cases:
            with self.subTest(gamma=gamma):
                self.model.basis_, self.model.gamma_ = basis, gamma
                with self.assertRaises(FloatingPointError):
                    self.original(self.model, query)
                with self.assertRaises(FloatingPointError):
                    self.model.decision_function(query)

    def test_nonfinite_scores_and_incompatible_coefficient_shape_still_fail(self):
        for field, values, error in [("dual_coef_", np.full((1, 12), np.nan), FloatingPointError),
                                      ("intercept_", np.array([np.inf]), FloatingPointError),
                                      ("dual_coef_", np.ones((1, 11)), ValueError)]:
            with self.subTest(field=field, error=error):
                model = copy.deepcopy(self.model)
                setattr(model, field, values)
                with self.assertRaises(error):
                    self.original(model, self.query)
                with self.assertRaises(error):
                    model.decision_function(self.query)

    def test_prediction_has_no_persistent_state_and_pickle_roundtrips(self):
        self.model.prediction_batch_size = 4
        before = pickle.dumps(self.model)
        expected = self.model.decision_function(self.query)
        self.model.decision_function(self.query[::-1])
        self.assertEqual(pickle.dumps(self.model), before)
        restored = pickle.loads(before)
        self.assert_bits(restored.decision_function(self.query), expected)

    def test_other_kernel_routes_are_unchanged(self):
        cases = [dict(kernel="linear"), dict(kernel="poly", gamma=0.4),
                 dict(kernel=linear_kernel), dict(kernel="rbf", gamma=0.0)]
        for settings in cases:
            with self.subTest(settings=settings):
                model = KernelSGDDWD(max_epochs=2, random_state=8, **settings).fit(self.X, self.y)
                with patch("two_stage_sgd_dwd._estimators.rbf_prediction_scores",
                           side_effect=AssertionError("Unrelated kernel route changed")):
                    self.assert_bits(model.decision_function(self.query), self.original(model, self.query))

    def test_training_validation_and_norm_never_use_prediction_helper(self):
        for batch_size in (1, 2, "full"):
            with self.subTest(batch_size=batch_size):
                checkpoints = []
                with patch("two_stage_sgd_dwd._estimators.rbf_prediction_scores",
                           side_effect=AssertionError("Training used prediction helper")):
                    model = KernelSGDDWD(gamma=0.4, max_epochs=2, min_epochs=0,
                                         validation_interval=1, batch_size=batch_size,
                                         random_state=8).fit(
                                             self.X, self.y, validation_data=(self.X, self.y),
                                             checkpoint_callback=lambda *args: checkpoints.append(args))
                    self.assertTrue(np.isfinite(model.rkhs_norm_squared()))
                    self.assertTrue(checkpoints)


if __name__ == "__main__":
    unittest.main()
