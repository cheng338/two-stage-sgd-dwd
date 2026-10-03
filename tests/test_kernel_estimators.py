"""Kernel API integration on tiny synthetic fixtures; no dataset experiments.

These tests are written for the generalized kernel API. Test execution is owned
by the serial verification harness.
"""

import copy
import pickle
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from two_stage_sgd_dwd import KernelSGDDWD, NumericalFailure, TwoStageDWDClassifier
from two_stage_sgd_dwd._core import rbf_kernel


def tiny_kernel_data():
    return (
        np.array([[-1.5, 0.2], [-0.7, -0.5], [0.2, 0.6],
                  [0.9, -0.3], [1.4, 0.4]]),
        np.array([-1, -1, 1, 1, 1]),
    )


def symmetric_selection_data():
    x = np.array([-8.0, -4.0, -1.0, -0.25, 0.25, 1.0, 4.0, 8.0])
    return np.column_stack((x, np.zeros_like(x))), np.where(x > 0, 1, -1)


def weighted_linear_matrix(X, Z, *, weights=(0.5, 1.25), offset=0.3):
    """Picklable matrix callable that cannot accept scalar-pair dispatch or gamma."""
    X, Z = np.asarray(X), np.asarray(Z)
    if X.ndim != 2 or Z.ndim != 2:
        raise AssertionError("The kernel callable must receive two feature matrices")
    return (X * np.asarray(weights)) @ Z.T + offset


def independent_loss(margins):
    values = np.empty_like(margins, dtype=float)
    lower = margins <= 0.5
    values[lower] = 1.0 - margins[lower]
    values[~lower] = 0.25 / margins[~lower]
    return values


def underlying_kernel(model):
    return model.kernel_estimator_ if isinstance(model, TwoStageDWDClassifier) else model


class GeneralKernelEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = tiny_kernel_data()

    def models(self, **parameters):
        settings = dict(max_epochs=2, batch_size="full", random_state=13)
        settings.update(parameters)
        return [
            KernelSGDDWD(**settings),
            TwoStageDWDClassifier(retention_fraction=1.0, **settings),
        ]

    def test_default_lambd_scale_gamma_and_preparation_diagnostics(self):
        # Compute the whole input variance independently of kernel resolution.
        variance = float(np.mean((self.X - self.X.mean()) ** 2))
        expected_gamma = 1.0 / (self.X.shape[1] * variance)
        for model in self.models():
            with self.subTest(estimator=type(model).__name__):
                self.assertEqual(model.lambd, 1e-6)
                self.assertNotIn("penalty_lambda", model.get_params())
                self.assertEqual(model.gamma, "scale")
                self.assertEqual(model.kernel, "rbf")
                model.fit(self.X, self.y)
                kernel = underlying_kernel(model)
                self.assertEqual(model.lambd_, 1e-6)
                self.assertEqual(kernel.lambd_, 1e-6)
                self.assertEqual(model.gamma, "scale")
                self.assertEqual(model.kernel_, "rbf")
                self.assertAlmostEqual(model.gamma_, expected_gamma, places=14)
                self.assertAlmostEqual(kernel.gamma_, expected_gamma, places=14)
                self.assertAlmostEqual(model.kernel_kws_["gamma"], expected_gamma, places=14)
                self.assertEqual(kernel.max_kernel_diagonal_, 1.0)
                self.assertEqual(kernel.learning_rate_B_, 8.000002)
                self.assertGreaterEqual(kernel.kernel_preparation_seconds_, 0.0)
                self.assertTrue(np.isfinite(kernel.kernel_preparation_seconds_))
                self.assertIsInstance(kernel.kernel_preparation_evaluation_count_, (int, np.integer))
                self.assertGreaterEqual(kernel.kernel_preparation_evaluation_count_, 0)
                self.assertTrue(kernel.kernel_validation_)

    def test_linear_and_polynomial_one_full_batch_step_match_manual_equations(self):
        cases = [
            ("linear", {}, None, self.X @ self.X.T),
            ("poly", {"degree": 2, "coef0": 0.7}, 0.4,
             (0.4 * (self.X @ self.X.T) + 0.7) ** 2),
        ]
        lambd = 0.06
        for name, keywords, expected_gamma, K in cases:
            with self.subTest(kernel=name):
                model = KernelSGDDWD(
                    kernel=name, kernel_kws=keywords, gamma=0.4,
                    lambd=lambd, batch_size="full", max_epochs=1,
                    random_state=23,
                ).fit(self.X, self.y)
                maximum_diagonal = float(np.diag(K).max())
                eta = 1.0 / (2.0 * lambd + 4.0 * (maximum_diagonal + 1.0))
                # All initial margins are zero, so q_i = -y_i. This is an
                # RKHS coefficient increment, not a Gram-multiplied increment.
                expected_a = eta * self.y / len(self.y)
                expected_b = eta * float(self.y.mean())
                assert_allclose(model.dual_coef_[0], expected_a, rtol=3e-15, atol=1e-16)
                assert_allclose(model.intercept_[0], expected_b, rtol=3e-15, atol=1e-16)
                self.assertEqual(model.n_updates_, 1)
                self.assertEqual(model.n_samples_seen_, len(self.y))
                self.assertEqual(model.gamma_, expected_gamma)
                self.assertAlmostEqual(model.max_kernel_diagonal_, maximum_diagonal, places=13)
                self.assertAlmostEqual(model.learning_rate_B_, 1.0 / eta, places=13)
                expected_scores = K @ expected_a + expected_b
                assert_allclose(model.decision_function(self.X), expected_scores, rtol=3e-14, atol=1e-15)
                expected_norm = float(expected_a @ K @ expected_a)
                self.assertAlmostEqual(model.rkhs_norm_squared(), expected_norm, places=14)
                expected_objective = float(independent_loss(self.y * expected_scores).mean()
                                           + lambd * expected_norm)
                self.assertAlmostEqual(model.objective(self.X, self.y), expected_objective, places=14)

    def test_lambd_multiplies_full_rkhs_norm_and_sets_learning_bound(self):
        for prototype in self.models(kernel="poly", kernel_kws={"degree": 2, "coef0": 0.5}, gamma=0.3):
            with self.subTest(estimator=type(prototype).__name__):
                model = clone(prototype).set_params(lambd=0.04).fit(self.X, self.y)
                kernel = underlying_kernel(model)
                self.assertEqual(model.lambd_, 0.04)
                self.assertEqual(kernel.lambd_, 0.04)
                gram = (0.3 * (kernel.basis_ @ kernel.basis_.T) + 0.5) ** 2
                query_kernel = (0.3 * (self.X @ kernel.basis_.T) + 0.5) ** 2
                coefficients = kernel.dual_coef_[0]
                expected_scores = query_kernel @ coefficients + kernel.intercept_[0]
                expected_norm = float(coefficients @ gram @ coefficients)
                expected_objective = (independent_loss(self.y * expected_scores).mean()
                                      + 0.04 * expected_norm)
                self.assertAlmostEqual(kernel.objective(self.X, self.y), expected_objective, places=14)
                self.assertAlmostEqual(kernel.learning_rate_B_,
                                       0.08 + 4.0 * (float(np.diag(gram).max()) + 1.0))

    def test_lambd_is_only_public_regularization_parameter(self):
        for estimator_class in (KernelSGDDWD, TwoStageDWDClassifier):
            with self.subTest(estimator=estimator_class.__name__):
                model = estimator_class(lambd=0.04)
                self.assertEqual(model.get_params()["lambd"], 0.04)
                self.assertNotIn("penalty_lambda", model.get_params())
                with self.assertRaises(TypeError):
                    estimator_class(penalty_lambda=0.08)
                with self.assertRaises(ValueError):
                    model.set_params(penalty_lambda=0.08)

    def test_clone_preserves_constructor_values_without_resolving_them(self):
        for model in self.models(kernel="poly", kernel_kws={"degree": 2, "coef0": 0.7},
                                 gamma="scale", lambd=0.02):
            with self.subTest(estimator=type(model).__name__):
                copied = clone(model)
                self.assertEqual(copied.get_params(), model.get_params())
                self.assertIsNot(copied.kernel_kws, model.kernel_kws)
                self.assertNotIn("penalty_lambda", copied.get_params())
                self.assertEqual(copied.gamma, "scale")
                copied.fit(self.X, self.y)
                self.assertEqual(copied.gamma, "scale")
                self.assertEqual(copied.lambd_, 0.02)
                self.assertEqual(copied.lambd, 0.02)

    def test_gamma_auto_and_scale_are_resolved_on_standalone_fit_inputs(self):
        for supplied, expected in [
            ("auto", 1.0 / self.X.shape[1]),
            ("scale", 1.0 / (self.X.shape[1] * self.X.var())),
        ]:
            with self.subTest(gamma=supplied):
                model = KernelSGDDWD(gamma=supplied, max_epochs=1, batch_size="full").fit(self.X, self.y)
                self.assertEqual(model.gamma_, expected)
                self.assertEqual(model.kernel_kws_["gamma"], expected)
                reference = KernelSGDDWD(gamma=expected, max_epochs=1, batch_size="full").fit(self.X, self.y)
                assert_array_equal(model.dual_coef_, reference.dual_coef_)
                assert_array_equal(model.decision_function(self.X), reference.decision_function(self.X))

    def test_zero_gamma_is_a_constant_rbf_kernel(self):
        model = KernelSGDDWD(gamma=0.0, max_epochs=3, batch_size="full").fit(self.X, self.y)
        queries = np.array([[-100.0, 23.0], [0.0, 0.0], [250.0, -90.0]])
        expected = np.full(len(queries), model.dual_coef_[0].sum() + model.intercept_[0])
        assert_allclose(model.decision_function(queries), expected, rtol=0, atol=1e-15)
        self.assertEqual(model.gamma_, 0.0)
        self.assertEqual(model.max_kernel_diagonal_, 1.0)
        self.assertAlmostEqual(model.rkhs_norm_squared(), float(model.dual_coef_[0].sum() ** 2), places=14)

    def test_matrix_callable_receives_kwargs_and_ignores_top_level_gamma(self):
        weights = np.array([0.5, 1.25])
        for model in self.models(kernel=weighted_linear_matrix,
                                 kernel_kws={"weights": weights, "offset": 0.3}, gamma=-123):
            with self.subTest(estimator=type(model).__name__):
                model.fit(self.X, self.y)
                kernel = underlying_kernel(model)
                self.assertIs(model.kernel_, weighted_linear_matrix)
                self.assertIsNone(model.gamma_)
                self.assertNotIn("gamma", model.kernel_kws_)
                K = weighted_linear_matrix(self.X, kernel.basis_, weights=weights, offset=0.3)
                expected = K @ kernel.dual_coef_[0] + kernel.intercept_[0]
                assert_allclose(model.decision_function(self.X), expected, rtol=1e-14, atol=1e-15)
                self.assertAlmostEqual(kernel.max_kernel_diagonal_,
                                       float(np.diag(weighted_linear_matrix(kernel.basis_, kernel.basis_,
                                                                            weights=weights, offset=0.3)).max()))
                loaded = pickle.loads(pickle.dumps(model))
                assert_array_equal(loaded.decision_function(self.X), model.decision_function(self.X))
                self.assertEqual(underlying_kernel(loaded).objective(self.X, self.y), kernel.objective(self.X, self.y))

    def test_fitted_kernel_is_frozen_against_constructor_and_metadata_mutation(self):
        for prototype in self.models():
            with self.subTest(estimator=type(prototype).__name__):
                original = {"weights": np.array([0.5, 1.25]), "offset": 0.3}
                model = prototype.set_params(kernel=weighted_linear_matrix, kernel_kws=original, lambd=0.04)
                model.fit(self.X, self.y)
                kernel = underlying_kernel(model)
                expected_scores = model.decision_function(self.X)
                expected_objective = kernel.objective(self.X, self.y)
                original["weights"][:] = 19.0
                original["offset"] = 25.0
                assert_array_equal(model.kernel_kws_["weights"], [0.5, 1.25])
                model.kernel_kws_["weights"][:] = 7.0
                kernel.kernel_kws_["offset"] = 11.0
                model.set_params(kernel="rbf", kernel_kws={"invalid": True}, gamma=22.0,
                                 lambd=15.0)
                if model is not kernel:
                    kernel.set_params(kernel="linear", gamma=45.0, lambd=33.0)
                self.assertEqual(model.lambd_, 0.04)
                self.assertEqual(kernel.lambd_, 0.04)
                assert_array_equal(model.decision_function(self.X), expected_scores)
                self.assertEqual(kernel.objective(self.X, self.y), expected_objective)

    def test_lambd_parameter_changes_take_effect_only_on_refit(self):
        for model in self.models(lambd=0.02):
            with self.subTest(estimator=type(model).__name__):
                model.fit(self.X, self.y)
                kernel = underlying_kernel(model)
                old_scores = model.decision_function(self.X)
                old_objective = kernel.objective(self.X, self.y)
                model.set_params(lambd=0.08)
                self.assertEqual(model.lambd_, 0.02)
                self.assertEqual(kernel.lambd_, 0.02)
                assert_array_equal(model.decision_function(self.X), old_scores)
                self.assertEqual(kernel.objective(self.X, self.y), old_objective)
                model.fit(self.X, self.y)
                updated_kernel = underlying_kernel(model)
                self.assertEqual(model.lambd_, 0.08)
                self.assertEqual(updated_kernel.lambd_, 0.08)
                self.assertAlmostEqual(updated_kernel.learning_rate_B_, 8.16)
                expected = clone(model).fit(self.X, self.y)
                assert_array_equal(model.decision_function(self.X), expected.decision_function(self.X))

    def test_explicit_rbf_single_observation_parameters_preserve_legacy_arithmetic(self):
        gamma, historical_penalty, seed, epochs = 0.7, 0.017, 29, 3
        coefficients = np.zeros(len(self.y))
        intercept = 0.0
        generator = np.random.default_rng(seed)
        for t in range(epochs * len(self.y)):
            index = int(generator.integers(len(self.y)))
            eta = (1.0 / (historical_penalty + 8.0)) * (1.0 + t / len(self.y)) ** (-0.75)
            column = rbf_kernel(self.X, self.X[index:index + 1], gamma).ravel()
            margin = self.y[index] * (float(np.dot(coefficients, column)) + intercept)
            derivative = -1.0 if margin <= 0.5 else (-0.25 / margin) / margin
            label_gradient = derivative * self.y[index]
            coefficients = (1.0 - eta * historical_penalty) * coefficients
            coefficients[index] -= eta * label_gradient
            intercept -= eta * label_gradient
        model = KernelSGDDWD(gamma=gamma, lambd=historical_penalty / 2.0, random_state=seed,
                            batch_size=1, sampling="with_replacement", max_epochs=epochs).fit(self.X, self.y)
        assert_array_equal(model.dual_coef_[0], coefficients)
        self.assertEqual(model.intercept_[0], intercept)


class TwoStageKernelResolutionTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = symmetric_selection_data()

    def test_default_scale_gamma_is_computed_before_retention(self):
        model = TwoStageDWDClassifier(retention_fraction=0.5,
                                      batch_size="full", max_epochs=1, random_state=9).fit(self.X, self.y)
        kernel = model.kernel_estimator_
        full_variance = float(np.mean((self.X - self.X.mean()) ** 2))
        retained_variance = float(np.mean((kernel.basis_ - kernel.basis_.mean()) ** 2))
        expected = 1.0 / (self.X.shape[1] * full_variance)
        retained_only = 1.0 / (self.X.shape[1] * retained_variance)
        self.assertNotEqual(expected, retained_only)
        self.assertEqual(model.gamma_, expected)
        self.assertEqual(kernel.gamma_, expected)
        self.assertEqual(kernel.gamma, expected)
        assert_array_equal(np.sort(model.selected_indices_), [2, 3, 4, 5])
        numeric = TwoStageDWDClassifier(retention_fraction=0.5, gamma=expected,
                                        batch_size="full", max_epochs=1, random_state=9).fit(self.X, self.y)
        assert_array_equal(model.selected_indices_, numeric.selected_indices_)
        assert_array_equal(model.decision_function(self.X), numeric.decision_function(self.X))

    def test_kernel_diagonal_bound_uses_actual_retained_basis(self):
        model = TwoStageDWDClassifier(retention_fraction=0.5, kernel="linear",
                                      batch_size="full", max_epochs=1, random_state=9,
                                      lambd=0.02).fit(self.X, self.y)
        kernel = model.kernel_estimator_
        retained_diagonal = float(np.sum(kernel.basis_ ** 2, axis=1).max())
        self.assertLess(retained_diagonal, float(np.sum(self.X ** 2, axis=1).max()))
        self.assertEqual(kernel.max_kernel_diagonal_, retained_diagonal)
        self.assertEqual(kernel.learning_rate_B_, 0.04 + 4.0 * (retained_diagonal + 1.0))
        self.assertIsNone(model.gamma_)

    def test_grid_search_clones_kernel_parameters_and_scales_each_fold(self):
        seen_scales = []

        def scoring(estimator, X_test, y_test):
            basis = underlying_kernel(estimator).basis_
            expected = 1.0 / (basis.shape[1] * basis.var())
            self.assertEqual(estimator.gamma_, expected)
            seen_scales.append(estimator.gamma_)
            return estimator.score(X_test, y_test)

        for prototype in [
            KernelSGDDWD(gamma="scale", max_epochs=1, batch_size="full", random_state=11),
            TwoStageDWDClassifier(gamma="scale", retention_fraction=1.0,
                                 max_epochs=1, batch_size="full", random_state=11),
        ]:
            with self.subTest(estimator=type(prototype).__name__):
                search = GridSearchCV(
                    prototype,
                    [{"kernel": ["rbf"], "lambd": [0.02]},
                     {"kernel": ["poly"], "kernel_kws": [{"degree": 2, "coef0": 1.0}],
                      "lambd": [0.04]}],
                    cv=StratifiedKFold(2, shuffle=True, random_state=3),
                    scoring=scoring, error_score="raise", n_jobs=1,
                ).fit(self.X, self.y)
                self.assertEqual(search.predict(self.X).shape, self.y.shape)
                self.assertTrue(np.isfinite(search.cv_results_["mean_test_score"]).all())
                self.assertEqual(search.best_estimator_.gamma_,
                                 1.0 / (self.X.shape[1] * self.X.var()))
        self.assertEqual(len(seen_scales), 8)


class KernelFailureAndRefitTests(unittest.TestCase):
    def test_invalid_kernel_configuration_cannot_leave_an_old_predictor(self):
        X, y = tiny_kernel_data()
        invalid = [
            {"kernel": "precomputed"},
            {"kernel": "rbf", "kernel_kws": {"degree": 2}},
            {"kernel": "rbf", "gamma": -0.1},
            {"kernel": "rbf", "gamma": "not-a-gamma-rule"},
            {"kernel": "poly", "kernel_kws": {"degree": 2, "coef0": -1.0}},
            {"lambd": 0.0},
            {"lambd": -0.1},
            {"lambd": None},
        ]
        for estimator_class in (KernelSGDDWD, TwoStageDWDClassifier):
            settings = {"retention_fraction": 1.0} if estimator_class is TwoStageDWDClassifier else {}
            for parameters in invalid:
                with self.subTest(estimator=estimator_class.__name__, parameters=parameters):
                    model = estimator_class(max_epochs=1, batch_size="full", **settings).fit(X, y)
                    model.set_params(**copy.deepcopy(parameters))
                    with self.assertRaises((ValueError, TypeError, NumericalFailure)):
                        model.fit(X, y)
                    with self.assertRaises(NotFittedError):
                        model.decision_function(X)
                    with self.assertRaises(NotFittedError):
                        model.predict(X)
                    model.set_params(kernel="linear", kernel_kws=None, gamma="auto",
                                     lambd=0.03)
                    model.fit(X, y)
                    self.assertEqual(model.kernel_, "linear")
                    self.assertEqual(model.lambd_, 0.03)
                    self.assertIsNone(model.gamma_)
                    self.assertTrue(np.isfinite(model.decision_function(X)).all())

    def test_indefinite_sigmoid_is_rejected_despite_positive_kernel_diagonal(self):
        X, y = np.array([[1.0], [2.0]]), np.array([-1, 1])
        # tanh(0.5)*tanh(2) - tanh(1)^2 < 0: this 2x2 symmetric
        # training Gram matrix has a negative eigenvalue although K_ii > 0.
        for model in [
            KernelSGDDWD(kernel="sigmoid", gamma=0.5, kernel_kws={"coef0": 0.0},
                         max_epochs=1, batch_size="full"),
            TwoStageDWDClassifier(kernel="sigmoid", gamma=0.5, kernel_kws={"coef0": 0.0},
                                  retention_fraction=1.0, max_epochs=1, batch_size="full"),
        ]:
            with self.subTest(estimator=type(model).__name__):
                with self.assertRaises((ValueError, NumericalFailure)):
                    model.fit(X, y)
                with self.assertRaises(NotFittedError):
                    model.predict(X)


if __name__ == "__main__":
    unittest.main()
