"""Numerical report recovery, independent references and read-only contracts."""

from decimal import Decimal, localcontext
import json
import math
import pickle
import random
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from threadpoolctl import threadpool_limits

from two_stage_sgd_dwd import KernelSGDDWD, LinearSGDDWD, optimization_diagnostics
from two_stage_sgd_dwd import _core, _kernels, diagnostics


def snapshot(X, *, kernel=True, gamma=1e-30, lambd=1e-40, intercept=0.0):
    """Construct a known feasible state to isolate reporting from optimization."""
    model = KernelSGDDWD() if kernel else LinearSGDDWD()
    model._fitted_ = True
    model.n_samples_fit_, model.n_features_in_ = X.shape
    model.classes_ = np.array([-1, 1])
    model.intercept_ = np.array([intercept])
    if kernel:
        model.basis_ = X.copy()
        model.dual_coef_ = np.zeros((1, len(X)))
        model.lambd_ = lambd
        model._kernel_spec_ = _kernels.resolve_kernel(X, gamma=gamma)
    else:
        model.coef_ = np.zeros((1, X.shape[1]))
    return model


def decimal_quadratic(X, vector, gamma):
    """Evaluate exact binary64 inputs using independent 160-digit arithmetic."""
    with localcontext() as context:
        context.prec = 160
        width = Decimal.from_float(float(gamma))
        v = [Decimal.from_float(float(value)) for value in vector]
        rows = [[Decimal.from_float(float(value)) for value in row] for row in X]
        result = Decimal(0)
        for i, left in enumerate(rows):
            for j, right in enumerate(rows):
                squared = sum(((a - b) ** 2 for a, b in zip(left, right)), Decimal(0))
                result += v[i] * v[j] * (-width * squared).exp()
        return result


class DiagnosticRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = threadpool_limits(limits=1)
        cls._threads.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._threads.__exit__(None, None, None)

    def test_six_finite_means_survive_overflow_of_the_loss_sum(self):
        for n in (4, 32, 128):
            X = np.zeros((n, 2))
            y = np.r_[-np.ones(n // 2), np.ones(n // 2)]
            for kernel in (False, True):
                model = snapshot(X, kernel=kernel, lambd=0.2, intercept=-1e308)
                before = pickle.dumps(model, protocol=5)
                with self.subTest(n=n, kernel=kernel):
                    report = optimization_diagnostics(model, X, y)
                    self.assertEqual(report["objective"], 5e307)
                    self.assertTrue(math.isfinite(report["numerical_gap_estimate"]))
                    self.assertEqual(pickle.dumps(model, protocol=5), before)
                    self.assertFalse(report["formal_certificate"])
                    json.dumps(report, allow_nan=False)

    def test_mean_fallback_preserves_ordinary_compensated_arithmetic(self):
        margins = np.array([-1.0, 0.5, np.nextafter(0.5, np.inf), 2.0, 1e308])
        expected_losses = [1.0 - x if x <= 0.5 else 0.25 / x for x in margins]
        with patch.object(diagnostics, "_mean_finite_losses", side_effect=AssertionError("fallback")):
            mean, alpha = diagnostics._loss_and_alpha(margins)
        self.assertEqual(mean, math.fsum(expected_losses) / len(margins))
        assert_array_equal(alpha[:2], [1.0, 1.0])
        for value in (np.inf, -np.inf, np.nan):
            with self.subTest(value=value), self.assertRaisesRegex(FloatingPointError, "Nonfinite margins"):
                diagnostics._loss_and_alpha(np.array([value]))

    def test_near_constant_dual_uses_nonzero_contrast_instead_of_rounded_ones(self):
        X, y = np.array([[-1.0], [1.0]]), np.array([-1, 1])
        for gamma in (1e-30, 1e-16, 1e-9):
            model = snapshot(X, gamma=gamma, lambd=gamma * 1e-10)
            with self.subTest(gamma=gamma):
                report = optimization_diagnostics(model, X, y, block_size=1)
                reference = float(decimal_quadratic(X, np.array([-0.5, 0.5]), gamma))
                self.assertGreater(report["raw_dual_rkhs_squared"], 0.0)
                self.assertLessEqual(abs(report["raw_dual_rkhs_squared"] - reference),
                                     4 * np.finfo(float).eps * reference)
                self.assertEqual(report["numerical_dual_bound_estimate"], 0.0)
                self.assertEqual(report["dual_bound_source"], "zero_dual_candidate")
                self.assertTrue(report["dual_bound_roundoff_warning"])
                self.assertFalse(report["formal_certificate"])

    def test_asymmetric_dual_quadratic_matches_same_vector_high_precision_reference(self):
        X = np.array([[-1.3, 0.4], [-0.2, -0.1], [0.4, 0.2], [0.8, -0.3], [1.7, 0.1]])
        y = np.array([-1, -1, -1, 1, 1])
        for gamma, lambd in ((1e-30, 1e-40), (1e-16, 1e-18), (1e-9, 1e-9)):
            model = snapshot(X, gamma=gamma, lambd=lambd, intercept=0.7)
            model.dual_coef_[0] = np.array([0.2, -0.2, 0.3, -0.4, 0.1])
            captured = {}
            original = diagnostics._balanced_alpha

            def capture(alpha, signed):
                result = original(alpha, signed)
                captured["q"] = result[1].copy()
                return result

            before = pickle.dumps(model, protocol=5)
            with self.subTest(gamma=gamma), patch.object(diagnostics, "_balanced_alpha", side_effect=capture):
                report = optimization_diagnostics(model, X, y, block_size=2)
            reference = float(decimal_quadratic(X, captured["q"], gamma))
            self.assertLessEqual(abs(report["raw_dual_rkhs_squared"] - reference),
                                 32 * np.finfo(float).eps * max(abs(reference), 1e-300))
            self.assertEqual(pickle.dumps(model, protocol=5), before)

    def test_recovery_retains_complete_report_schema_and_two_pass_accounting(self):
        X = np.linspace(-1, 1, 9)[:, None]
        y = np.where(X[:, 0] < 0, -1, 1)
        ordinary = optimization_diagnostics(snapshot(X, gamma=0.7, lambd=0.2), X, y)
        model = snapshot(X)
        for block_size in (1, 4, 9, 32):
            calls = []
            original = diagnostics.cdist

            def counted(left, right, **kwargs):
                calls.append((len(left), len(right)))
                return original(left, right, **kwargs)

            with self.subTest(block_size=block_size), patch.object(diagnostics, "cdist", side_effect=counted):
                report = optimization_diagnostics(model, X, y, block_size=block_size)
            self.assertEqual(set(report), set(ordinary))
            self.assertEqual(len(calls), 2 * math.ceil(len(X) / block_size))
            self.assertEqual(sum(left * right for left, right in calls), 2 * len(X) ** 2)
            self.assertEqual(report["n_kernel_values"], 2 * len(X) ** 2)
            self.assertEqual(report["kernel_passes"], 2)
            self.assertEqual(report["max_block_elements"], min(len(X), block_size) * len(X))
            self.assertEqual(report["n_samples"], len(X))
            self.assertEqual(report["n_features"], 1)
            self.assertEqual(report["lambd"], model.lambd_)
            self.assertTrue(report["fitting_rows_verified_against_model"])
            self.assertIn("original fitting labels", report["fitting_labels_requirement"])
            json.dumps(report, allow_nan=False)

    def test_custom_or_modified_kernel_evaluation_is_not_bypassed(self):
        X, y = np.array([[-1.0], [1.0]]), np.array([-1, 1])
        model = snapshot(X)
        original = _kernels.KernelSpec.evaluate
        with patch.object(_kernels.KernelSpec, "evaluate", autospec=True, side_effect=original) as method:
            optimization_diagnostics(model, X, y, block_size=1)
            self.assertEqual(method.call_count, 4)
        with patch.object(_kernels, "rbf_kernel", wraps=_kernels.rbf_kernel) as kernel:
            optimization_diagnostics(model, X, y, block_size=1)
            self.assertEqual(kernel.call_count, 4)
        for values in (np.full((2, 2), np.nan), -np.ones((2, 2)), np.ones((2, 3))):
            with self.subTest(shape=values.shape), patch.object(_kernels.KernelSpec, "evaluate", return_value=values):
                with self.assertRaises((ValueError, FloatingPointError)):
                    optimization_diagnostics(model, X, y)

    def test_modified_core_distance_hook_uses_original_kernel_path(self):
        X, y = np.array([[-1.0], [1.0]]), np.array([-1, 1])
        model = snapshot(X)
        before = pickle.dumps(model, protocol=5)
        with patch.object(_core, "cdist", wraps=_core.cdist) as distance:
            self.assertIsNone(diagnostics._near_constant_rbf_width(model._kernel_spec_, X))
            with patch.object(diagnostics, "cdist", side_effect=AssertionError("contrast bypass")):
                report = optimization_diagnostics(model, X, y, block_size=1)
            self.assertEqual(distance.call_count, 4)
        self.assertEqual(report["n_kernel_values"], 8)
        self.assertEqual(pickle.dumps(model, protocol=5), before)

    def test_ordinary_and_zero_width_arithmetic_is_unchanged(self):
        X, y = np.array([[-1.0, 0.2], [-0.3, 0.4], [0.2, -0.1], [1.0, 0.8]]), np.array([-1, -1, 1, 1])
        for name, gamma in (("rbf", 0.7), ("rbf", 0.0), ("cosine", 0.7), ("linear", 0.7)):
            model = snapshot(X, gamma=gamma, lambd=0.2)
            model._kernel_spec_ = _kernels.resolve_kernel(X, name, gamma)
            model.dual_coef_[0] = np.array([0.13, -0.17, 0.23, -0.11])
            report = optimization_diagnostics(model, X, y, block_size=3)
            with patch.object(diagnostics, "_near_constant_rbf_width", return_value=None):
                reference = optimization_diagnostics(model, X, y, block_size=3)
            report.pop("elapsed_seconds")
            reference.pop("elapsed_seconds")
            with self.subTest(kernel=name, gamma=gamma):
                self.assertEqual(report, reference)

    def test_fitted_states_rng_inputs_and_error_policy_remain_unchanged(self):
        X = np.array([[-1.0, 0.2], [-0.3, 0.4], [0.2, -0.1], [1.0, 0.8]])
        y = np.array([-1, -1, 1, 1])
        for batch_size in (1, "full"):
            model = KernelSGDDWD(gamma=1e-30, lambd=1e-40, max_epochs=20,
                batch_size=batch_size, stopping="fixed", optimizer_backend="python", random_state=71).fit(X, y)
            for mode in ("ignore", "warn", "raise", "call", "print", "log"):
                before = pickle.dumps(model, protocol=5)
                numpy_state, python_state = pickle.dumps(np.random.get_state()), random.getstate()
                saved_X, saved_y = X.copy(), y.copy()
                with self.subTest(batch_size=batch_size, policy=mode), np.errstate(all=mode):
                    policy = np.geterr().copy()
                    optimization_diagnostics(model, X, y, block_size=3)
                    self.assertEqual(np.geterr(), policy)
                self.assertEqual(pickle.dumps(model, protocol=5), before)
                self.assertEqual(pickle.dumps(np.random.get_state()), numpy_state)
                self.assertEqual(random.getstate(), python_state)
                assert_array_equal(X, saved_X)
                assert_array_equal(y, saved_y)

    def test_guard_failures_keep_original_errors_and_invalid_gap_is_not_relaxed(self):
        X, y = np.array([[-1.0], [1.0]]), np.array([-1, 1])
        model = snapshot(X)
        for invalid in (X[::-1], X[:1], X.astype(complex) + 1j):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                optimization_diagnostics(model, invalid, y)
        for gamma in (0.0, -1.0, np.inf, np.nan, True):
            spec = _kernels.KernelSpec("rbf", gamma, {})
            self.assertIsNone(diagnostics._near_constant_rbf_width(spec, X))
        for kws in ({"gamma": 0.1}, {"gamma": True}, {"unexpected": 1.0}):
            spec = _kernels.KernelSpec("rbf", 1e-30, kws)
            self.assertIsNone(diagnostics._near_constant_rbf_width(spec, X))
        large = np.array([[-1e308], [1e308]])
        self.assertIsNone(diagnostics._near_constant_rbf_width(model._kernel_spec_, large))
        with self.assertRaisesRegex(FloatingPointError, "Negative numerical gap"):
            diagnostics._result("test", 0.0, np.ones(2), np.ones(2), np.array([0.5, -0.5]),
                                0.0, 1.0, 1.0, 0.0, 0.0, 0.0)

    def test_roundoff_warning_includes_zero_and_positive_quadratics(self):
        X, y = np.array([[-1.0], [1.0]]), np.array([-1, 1])
        for gamma in (0.0, 1e-30):
            report = optimization_diagnostics(snapshot(X, gamma=gamma), X, y)
            with self.subTest(gamma=gamma):
                self.assertGreaterEqual(report["raw_dual_rkhs_squared"], 0.0)
                self.assertTrue(report["dual_bound_roundoff_warning"])
                self.assertIn("need not imply an actual error", report["roundoff_qualification"])
                self.assertFalse(report["formal_certificate"])


if __name__ == "__main__":
    unittest.main()
