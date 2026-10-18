"""Independent post-fit optimization checks and read-only API integration."""

import copy
import json
import math
import pickle
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

from two_stage_sgd_dwd import (KernelSGDDWD, LinearSGDDWD,
                               TwoStageDWDClassifier, optimization_diagnostics)
from two_stage_sgd_dwd import _estimators
from two_stage_sgd_dwd.diagnostics import _balanced_alpha, _result


def zero_kernel(X, Y):
    return np.zeros((len(X), len(Y)))


def independent_loss(u):
    return np.array([1.0 - z if z <= 0.5 else 0.25 / z for z in u])


def fitted_linear(X, y, w=None, b=0.0, **kwargs):
    model = LinearSGDDWD(max_epochs=1, batch_size="full", stopping="fixed",
                         random_state=17, optimizer_backend="python", **kwargs).fit(X, y)
    model.coef_ = np.zeros((1, X.shape[1])) if w is None else np.array([w], dtype=float)
    model.intercept_ = np.array([b], dtype=float)
    return model


def fitted_kernel(X, y, a=None, b=0.0, kernel="rbf", lambd=0.2):
    model = KernelSGDDWD(max_epochs=1, batch_size="full", stopping="fixed",
                         random_state=17, optimizer_backend="python",
                         prediction_backend="direct", kernel=kernel,
                         gamma=0.7, lambd=lambd).fit(X, y)
    model.dual_coef_ = np.zeros((1, len(X))) if a is None else np.array([a], dtype=float)
    model.intercept_ = np.array([b], dtype=float)
    return model


class OptimizationDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = threadpool_limits(limits=1)
        cls._threads.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._threads.__exit__(None, None, None)

    def setUp(self):
        self.X = np.array([[-1., 0.2], [-0.3, 0.8], [0.4, -0.3], [1.2, 0.5]])
        self.y = np.array([-1, -1, 1, 1])

    def assert_compact(self, report):
        self.assertNotIn("dual_alpha", report)
        self.assertNotIn("dual_q", report)
        self.assertFalse(report["formal_certificate"])
        json.dumps(report, allow_nan=False)

    def test_overflowing_diagnostic_sum_preserves_a_usable_fitted_state(self):
        # The loss mean is representable even though its diagnostic sum is not.
        # An unavailable optional diagnostic must not invalidate the model.
        model = fitted_linear(self.X, self.y, b=-1e308)
        predictions = model.predict(self.X)
        self.assertTrue(np.isfinite(model.objective(self.X, self.y)))
        before = pickle.dumps(model, protocol=5)
        with self.assertRaisesRegex(FloatingPointError, "diagnostic sum"):
            optimization_diagnostics(model, self.X, self.y)
        self.assertEqual(pickle.dumps(model, protocol=5), before)
        assert_array_equal(model.predict(self.X), predictions)
        self.assertTrue(model.fit_usable_)

    def test_linear_balanced_zero_analytic(self):
        X = np.zeros((4, 3))
        r = optimization_diagnostics(fitted_linear(X, self.y), X, self.y)
        self.assertEqual(r["objective"], 1.)
        self.assertEqual(r["numerical_dual_bound_estimate"], 1.)
        self.assertEqual(r["numerical_gap_estimate"], 0.)
        self.assertEqual(r["stationarity_residual"], 0.)
        self.assert_compact(r)

    def test_kernel_balanced_zero_analytic(self):
        for name in (zero_kernel, "rbf"):
            with self.subTest(kernel=name):
                X = np.zeros_like(self.X)
                model = fitted_kernel(X, self.y, kernel=name)
                r = optimization_diagnostics(model, X, self.y, block_size=2)
                self.assertEqual(r["objective"], 1.)
                self.assertEqual(r["numerical_gap_estimate"], 0.)
                self.assertEqual(r["stationarity_residual"], 0.)
                self.assertEqual(r["n_kernel_values"], 32)
                self.assertEqual(r["max_block_elements"], 8)

    def test_imbalanced_constant_optimum_has_free_intercept(self):
        X, y = np.zeros((4, 2)), np.array([-1, 1, 1, 1])
        b = math.sqrt(3.) / 2.
        expected = 0.25 + math.sqrt(3.) / 4.
        models = [fitted_linear(X, y, b=b), fitted_kernel(X, y, b=b, kernel=zero_kernel),
                  fitted_kernel(X, y, b=b)]
        for model in models:
            with self.subTest(estimator=type(model).__name__, kernel=getattr(model, "kernel", None)):
                r = optimization_diagnostics(model, X, y)
                self.assertAlmostEqual(r["objective"], expected, places=14)
                self.assertAlmostEqual(r["numerical_dual_bound_estimate"], expected, places=14)
                self.assertLess(abs(r["numerical_gap_estimate"]), 1e-14)
                self.assertLess(r["stationarity_residual"], 1e-14)

    def test_nonoptimal_zero_state_does_not_have_zero_gap(self):
        X, y = np.array([[-0.1], [0.1]]), np.array([-1, 1])
        r = optimization_diagnostics(fitted_linear(X, y), X, y)
        self.assertAlmostEqual(r["objective"], 1.)
        self.assertAlmostEqual(r["numerical_dual_bound_estimate"], .9)
        self.assertAlmostEqual(r["numerical_gap_estimate"], .1)
        self.assertAlmostEqual(r["stationarity_residual"], .1)

    def test_linear_interior_finite_difference(self):
        X = self.X * 0.2
        model = fitted_linear(X, self.y, w=[0.1, -0.05], b=0.7)
        theta = np.r_[model.coef_[0], model.intercept_[0]]
        fun = lambda z: np.mean(independent_loss(self.y * (X @ z[:-1] + z[-1])))
        h = 1e-6
        gradient = np.array([(fun(theta + np.eye(3)[j]*h) - fun(theta - np.eye(3)[j]*h))/(2*h)
                             for j in range(3)])
        r = optimization_diagnostics(model, X, self.y)
        self.assertLess(np.linalg.norm(theta[:-1] - gradient[:-1]), 1.)
        self.assertAlmostEqual(r["free_intercept_derivative"], gradient[-1], places=9)
        self.assertAlmostEqual(r["stationarity_residual"], np.linalg.norm(gradient), places=9)

    def test_linear_boundary_uses_projected_gradient(self):
        X, y = np.array([[-0.1], [0.1]]), np.array([-1, 1])
        r = optimization_diagnostics(fitted_linear(X, y, w=[1.0]), X, y)
        self.assertEqual(r["projected_gradient_norm"], 0.)
        self.assertAlmostEqual(r["objective"], .9)
        self.assertAlmostEqual(r["numerical_gap_estimate"], 0.)

    def test_kernel_finite_difference_and_independent_dual(self):
        model = fitted_kernel(self.X, self.y, a=[.13, -.17, .23, -.11], b=.68)
        K = np.exp(-.7 * ((self.X[:, None]-self.X[None, :])**2).sum(axis=2))
        theta = np.r_[model.dual_coef_[0], model.intercept_[0]]
        fun = lambda z: np.mean(independent_loss(self.y*(K@z[:-1]+z[-1]))) + .2*z[:-1]@K@z[:-1]
        h = 1e-6
        gradient = np.array([(fun(theta + np.eye(5)[j]*h) - fun(theta - np.eye(5)[j]*h))/(2*h)
                             for j in range(5)])
        margins = self.y * (K@theta[:-1]+theta[-1])
        alpha = np.array([1. if m <= .5 else .25/m**2 for m in margins])
        t = .4*theta[:-1]-self.y*alpha/len(self.y)
        assert_allclose(K@t, gradient[:-1], rtol=1e-8, atol=1e-9)
        r = optimization_diagnostics(model, self.X, self.y, block_size=3)
        self.assertAlmostEqual(r["objective"], fun(theta), places=14)
        self.assertAlmostEqual(r["free_intercept_derivative"], gradient[-1], places=9)
        self.assertAlmostEqual(r["rkhs_gradient_norm"], math.sqrt(t@K@t), places=14)
        positive, negative = self.y > 0, self.y < 0
        positive_total, negative_total = alpha[positive].sum(), alpha[negative].sum()
        if positive_total > negative_total:
            alpha[positive] *= negative_total/positive_total
        else:
            alpha[negative] *= positive_total/negative_total
        q = self.y*alpha/len(self.y)
        dual = np.mean(np.sqrt(alpha))-q@K@q/(4*.2)
        self.assertAlmostEqual(r["dual_objective_before_balance_correction"], dual, places=14)
        self.assert_compact(r)

    def test_nullspace_coefficient_invariance(self):
        X = np.array([[-1.], [-1.], [1.], [1.]])
        a = np.array([.05, .1, -.03, .07])
        first = optimization_diagnostics(fitted_kernel(X, self.y, a=a), X, self.y)
        second = optimization_diagnostics(
            fitted_kernel(X, self.y, a=a+np.array([1., -1., 2., -2.])), X, self.y)
        for key in ("objective", "rkhs_norm_squared", "rkhs_gradient_norm", "numerical_dual_bound_estimate"):
            self.assertAlmostEqual(first[key], second[key], places=13)

    def test_duplicate_row_representation_invariance(self):
        a = np.array([.13, -.17, .23, -.11])
        first = optimization_diagnostics(fitted_kernel(self.X, self.y, a=a, b=.68),
                                         self.X, self.y, block_size=2)
        X, y = np.repeat(self.X, 3, axis=0), np.repeat(self.y, 3)
        second = optimization_diagnostics(fitted_kernel(X, y, a=np.repeat(a/3., 3), b=.68),
                                          X, y, block_size=2)
        for key in ("objective", "rkhs_norm_squared", "rkhs_gradient_norm", "numerical_dual_bound_estimate"):
            self.assertAlmostEqual(first[key], second[key], places=13)

    def test_block_sizes_preserve_results_and_bound_memory(self):
        model = fitted_kernel(self.X, self.y, a=[.13, -.17, .23, -.11], b=.68)
        for block_size in (1, 3, 4, 100):
            with self.subTest(block_size=block_size):
                r = optimization_diagnostics(model, self.X, self.y, block_size=block_size)
                reference = optimization_diagnostics(model, self.X, self.y, block_size=4)
                self.assertAlmostEqual(r["objective"], reference["objective"], places=14)
                self.assertAlmostEqual(r["stationarity_residual"], reference["stationarity_residual"], places=14)
                self.assertEqual(r["n_kernel_values"], 2*len(self.X)**2)
                self.assertEqual(r["max_block_elements"], min(block_size, len(self.X))*len(self.X))

    def test_actual_fits_preserve_model_predictions_inputs_and_global_rng(self):
        for constructor in (LinearSGDDWD, KernelSGDDWD):
            with self.subTest(estimator=constructor.__name__):
                model = constructor(max_epochs=3, random_state=41, optimizer_backend="python",
                                    stopping="fixed").fit(self.X, self.y)
                before = pickle.dumps(model, protocol=5)
                predictions = model.predict(self.X)
                scores = model.decision_function(self.X)
                X, y = self.X.copy(), self.y.copy()
                numpy_rng, python_rng = pickle.dumps(np.random.get_state()), random.getstate()
                report = optimization_diagnostics(model, X, y)
                self.assertEqual(pickle.dumps(model, protocol=5), before)
                self.assertEqual(pickle.dumps(np.random.get_state()), numpy_rng)
                self.assertEqual(random.getstate(), python_rng)
                assert_array_equal(model.predict(self.X), predictions)
                assert_array_equal(model.decision_function(self.X), scores)
                assert_array_equal(X, self.X)
                assert_array_equal(y, self.y)
                self.assertFalse(model.optimality_checked_)
                self.assertFalse(model.converged_)
                self.assertTrue(model.fit_usable_)
                self.assertAlmostEqual(report["objective"], model.objective(self.X, self.y), places=14)
                self.assert_compact(report)

    def test_string_label_mapping(self):
        labels = np.array(["cat", "cat", "dog", "dog"])
        for factory in (fitted_linear, fitted_kernel):
            original = optimization_diagnostics(factory(self.X, self.y), self.X, self.y)
            mapped = optimization_diagnostics(factory(self.X, labels), self.X, labels)
            self.assertEqual(original["objective"], mapped["objective"])
            self.assertEqual(original["stationarity_residual"], mapped["stationarity_residual"])

    def test_actual_two_stage_uses_retained_mean_for_each_batch(self):
        X = np.column_stack((np.linspace(-2, 2, 40), np.cos(np.arange(40))*.03))
        y = np.where(X[:, 0] < 0, "left", "right")
        for batch in (1, 5, "full"):
            with self.subTest(batch=batch):
                model = TwoStageDWDClassifier(retention_fraction=.3, max_epochs=20,
                    random_state=23, gamma=.6, lambd=.015, batch_size=batch,
                    optimizer_backend="python", stopping="fixed").fit(X, y)
                before = pickle.dumps(model, protocol=5)
                predictions = model.predict(X)
                r = optimization_diagnostics(model, X, y, block_size=3)
                idx = model.selected_indices_
                reference = optimization_diagnostics(model.kernel_estimator_, X[idx], y[idx], block_size=3)
                self.assertEqual(len(idx), 12)
                self.assertEqual(r["linear_stage"]["n_samples"], 40)
                self.assertEqual(r["kernel_stage"]["n_samples"], 12)
                for key in ("objective", "numerical_gap_estimate", "stationarity_residual"):
                    self.assertEqual(r["kernel_stage"][key], reference[key])
                self.assertAlmostEqual(reference["objective"],
                                       model.kernel_estimator_.objective(X[idx], y[idx]), places=14)
                self.assertEqual(pickle.dumps(model, protocol=5), before)
                assert_array_equal(model.predict(X), predictions)
                self.assert_compact(r["linear_stage"])
                self.assert_compact(r["kernel_stage"])

    def test_validation_selected_checkpoint_is_assessed_instead_of_terminal_state(self):
        for constructor, attribute in ((LinearSGDDWD, "coef_"), (KernelSGDDWD, "dual_coef_")):
            with self.subTest(estimator=constructor.__name__):
                model = constructor(max_epochs=4, batch_size="full", validation_interval=1,
                    random_state=7, stopping="fixed", optimizer_backend="python")
                losses = iter([.8, 1., .9, 1.1])
                with patch.object(_estimators, "dwd_loss", side_effect=lambda x: np.full(len(x), next(losses))):
                    model.fit(self.X, self.y, validation_data=(self.X, self.y))
                self.assertLess(model.returned_iteration_, model.n_updates_)
                before = pickle.dumps(model, protocol=5)
                r = optimization_diagnostics(model, self.X, self.y)
                self.assertAlmostEqual(r["objective"], model.objective(self.X, self.y), places=14)
                self.assertEqual(pickle.dumps(model, protocol=5), before)
                terminal = copy.deepcopy(model)
                setattr(terminal, attribute, terminal.final_state_["coefficients"][None, :].copy())
                terminal.intercept_ = np.array([terminal.final_state_["intercept"]])
                self.assertGreater(abs(r["objective"]-terminal.objective(self.X, self.y)), 1e-5)

    def test_diagnostic_failure_leaves_fitted_model_usable(self):
        model = KernelSGDDWD(max_epochs=2, random_state=11).fit(self.X, self.y)
        before = pickle.dumps(model, protocol=5)
        scores = model.decision_function(self.X)
        invalid_blocks = (np.full((4, 4), np.nan), -np.ones((4, 4)), np.zeros((4, 5)))
        for values in invalid_blocks:
            with self.subTest(shape=values.shape, first_value=values[0, 0]):
                with patch.object(type(model._kernel_spec_), "evaluate", return_value=values):
                    with self.assertRaises((ValueError, FloatingPointError)):
                        optimization_diagnostics(model, self.X, self.y)
                self.assertEqual(pickle.dumps(model, protocol=5), before)
                assert_array_equal(model.decision_function(self.X), scores)
        self.assertTrue(model.fit_usable_)
        self.assertFalse(model.optimality_checked_)

    def test_current_constructor_parameters_do_not_replace_fitted_kernel(self):
        model = KernelSGDDWD(max_epochs=2, random_state=11, gamma=.7, lambd=.2).fit(self.X, self.y)
        first = optimization_diagnostics(model, self.X, self.y)
        model.set_params(kernel="linear", gamma=20., lambd=100.)
        second = optimization_diagnostics(model, self.X, self.y)
        for key in ("objective", "lambd", "rkhs_gradient_norm", "numerical_gap_estimate"):
            self.assertEqual(first[key], second[key])

    def test_balance_preserves_box_and_mean_normalization(self):
        signed = np.array([-1., 1., 1., 1.])
        alpha, q, residual = _balanced_alpha(np.array([.7, .3, .9, 1.]), signed)
        self.assertTrue(np.all((alpha >= 0.) & (alpha <= 1.)))
        self.assertLess(abs(residual), 1e-15)
        assert_array_equal(q, signed*alpha/len(alpha))

    def test_nonzero_balance_requires_free_intercept_correction(self):
        # With X=0 and labels [-1,1,1,1], this box-feasible candidate is
        # unbalanced. Its uncorrected expression exceeds the known optimum.
        alpha = np.array([1., .5, .5, .5])
        q = np.array([-1., 1., 1., 1.])*alpha/4
        optimum = .25+math.sqrt(3.)/4
        uncorrected = np.mean(np.sqrt(alpha))
        self.assertGreater(uncorrected, optimum)
        r = _result("test", optimum, alpha, alpha, q, .125, 3.,
                    uncorrected, 0., 0., 0.)
        self.assertAlmostEqual(r["dual_balance_correction"], .375)
        self.assertAlmostEqual(r["numerical_dual_bound_estimate"], uncorrected-.375)
        self.assertAlmostEqual(r["numerical_gap_estimate"], optimum-uncorrected+.375)

    def test_large_finite_intercept_bound_avoids_ratio_overflow(self):
        X, y = np.array([[1.e100], [1.e100]]), np.array([-1, 1])
        model = fitted_kernel(X, y, kernel="linear", lambd=1.e-200)
        r = optimization_diagnostics(model, X, y)
        self.assertTrue(math.isfinite(r["optimal_intercept_absolute_bound"]))
        self.assertGreater(r["optimal_intercept_absolute_bound"], 1.e199)

    def test_unfitted_and_unsupported_types_are_rejected(self):
        for model in (LinearSGDDWD(), KernelSGDDWD(), TwoStageDWDClassifier()):
            with self.subTest(estimator=type(model).__name__), self.assertRaises(NotFittedError):
                optimization_diagnostics(model, self.X, self.y)
        for value in (None, object(), SimpleNamespace(_fitted_=True)):
            with self.subTest(value=type(value).__name__), self.assertRaises(TypeError):
                optimization_diagnostics(value, self.X, self.y)

    def test_invalid_rows_labels_and_parameters_do_not_mutate_fit(self):
        model = fitted_kernel(self.X, self.y)
        before = pickle.dumps(model, protocol=5)
        for X, y in ((self.X[::-1], self.y[::-1]), (self.X[:3], self.y[:3]),
                     (self.X[:, :1], self.y), (self.X, np.ones(4)),
                     (self.X, np.array([-1, -1, 1, 9])), (self.X, self.y[:, None]),
                     (self.X.astype(complex)+1j, self.y)):
            with self.subTest(shape=np.shape(X), labels=np.shape(y)), self.assertRaises(ValueError):
                optimization_diagnostics(model, X, y)
        self.assertEqual(pickle.dumps(model, protocol=5), before)
        for size in (0, -1, True, 1.5, "2"):
            with self.subTest(block_size=size), self.assertRaises(ValueError):
                optimization_diagnostics(model, self.X, self.y, block_size=size)
        self.assertEqual(pickle.dumps(model, protocol=5), before)

    def test_malformed_fitted_states_are_rejected(self):
        for attribute, value in (("coef_", np.array([[np.nan, 0.]])),
                                 ("coef_", np.array([[2., 0.]])),
                                 ("intercept_", np.array([np.inf]))):
            model = fitted_linear(self.X, self.y)
            setattr(model, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                optimization_diagnostics(model, self.X, self.y)
        for lambd in (0., -1., np.inf, True):
            model = fitted_kernel(self.X, self.y)
            model.lambd_ = lambd
            with self.subTest(lambd=lambd), self.assertRaises(ValueError):
                optimization_diagnostics(model, self.X, self.y)

    def test_malformed_two_stage_metadata_is_rejected(self):
        X = np.column_stack((np.linspace(-2, 2, 40), np.cos(np.arange(40))*.03))
        y = np.where(X[:, 0] < 0, -1, 1)
        original = TwoStageDWDClassifier(retention_fraction=.3, max_epochs=5,
            random_state=23, batch_size="full").fit(X, y)
        mutations = [lambda m: setattr(m, "selected_indices_", np.repeat(m.selected_indices_[0], 12)),
                     lambda m: setattr(m, "selected_indices_", np.array([-1]+m.selected_indices_[1:].tolist())),
                     lambda m: setattr(m, "n_retained_", 11),
                     lambda m: setattr(m.kernel_estimator_, "classes_", np.array([-1, 2])),
                     lambda m: setattr(m.kernel_estimator_, "basis_", m.kernel_estimator_.basis_[::-1].copy())]
        for index, mutation in enumerate(mutations):
            model = copy.deepcopy(original)
            mutation(model)
            before = pickle.dumps(model, protocol=5)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                optimization_diagnostics(model, X, y)
            self.assertEqual(pickle.dumps(model, protocol=5), before)

    def test_materially_negative_gap_is_rejected(self):
        with self.assertRaisesRegex(FloatingPointError, "Negative numerical gap"):
            _result("test", 0., np.ones(2), np.ones(2), np.array([.5, -.5]), 0., 1., 1., 0., 0., 0.)

    def test_tiny_negative_gap_remains_visible(self):
        bound = float(np.nextafter(1., np.inf))
        report = _result("test", 1., np.ones(2), np.ones(2), np.array([.5, -.5]),
                         0., 1., bound, 0., 0., 0.)
        self.assertEqual(report["numerical_gap_estimate"], 1.-bound)
        self.assertTrue(report["negative_gap_observed"])
        self.assertFalse(report["formal_certificate"])


if __name__ == "__main__":
    unittest.main()
