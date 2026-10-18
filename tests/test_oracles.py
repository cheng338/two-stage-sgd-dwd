"""Independent deterministic objective checks, never MNIST experiments.

Finite SGD iterates need not coincide with an optimizer. These tests compare
well-scaled tiny problems after a sufficient fixed update budget and report
an explicit, deliberately nonzero optimization-gap tolerance.
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from two_stage_sgd_dwd import KernelSGDDWD, LinearSGDDWD, TwoStageDWDClassifier


def independent_loss_and_derivative(margins):
    margins = np.asarray(margins, dtype=np.float64)
    loss = np.empty_like(margins)
    derivative = np.empty_like(margins)
    for i, margin in enumerate(margins):
        if margin <= 0.5:
            loss[i] = 1.0 - margin
            derivative[i] = -1.0
        else:
            loss[i] = 0.25 / margin
            derivative[i] = (-0.25 / margin) / margin
    return loss, derivative


def independent_rbf(X, Z, gamma):
    return np.exp(-gamma * ((X[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))


class IndependentObjectiveOracleTests(unittest.TestCase):
    def test_linear_constrained_sgd_approaches_independent_slsqp_oracle(self):
        X = np.array([[-2.0], [-1.0], [-0.3], [0.3], [1.0], [2.0]])
        y = np.array([-1, -1, -1, 1, 1, 1])

        def objective(parameters):
            margins = y * (X[:, 0] * parameters[0] + parameters[1])
            loss, derivative = independent_loss_and_derivative(margins)
            d = derivative * y
            return loss.mean(), np.array([np.mean(d * X[:, 0]), d.mean()])

        oracle = minimize(
            objective, np.zeros(2), jac=True, method="SLSQP",
            constraints={
                "type": "ineq", "fun": lambda p: 1.0 - p[0] ** 2,
                "jac": lambda p: np.array([-2.0 * p[0], 0.0]),
            },
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        self.assertTrue(oracle.success, oracle.message)
        self.assertLessEqual(abs(oracle.x[0]), 1.0 + 1e-10)
        model = LinearSGDDWD(max_epochs=600, random_state=12).fit(X, y)
        fitted = np.array([model.coef_[0, 0], model.intercept_[0]])
        gap = objective(fitted)[0] - oracle.fun
        self.assertGreaterEqual(gap, -1e-8)
        self.assertLess(gap, 0.025, f"finite-budget constrained objective gap={gap}")
        self.assertLessEqual(abs(fitted[0]), 1.0 + 1e-14)
        assert_allclose(model.predict(X), y)

    def test_kernel_rkhs_objective_approaches_independent_bfgs_oracle(self):
        X = np.array([[-2.0], [-1.0], [-0.3], [0.3], [1.0], [2.0]])
        y = np.array([-1, -1, -1, 1, 1, 1])
        gamma, lambd = 0.7, 0.04
        K = independent_rbf(X, X, gamma)
        self.assertFalse(np.allclose(K, np.eye(len(X))))

        def objective(parameters):
            coefficients, intercept = parameters[:-1], parameters[-1]
            loss, derivative = independent_loss_and_derivative(y * (K @ coefficients + intercept))
            d = derivative * y
            value = loss.mean() + lambd * coefficients @ K @ coefficients
            gradient = np.r_[K @ (d / len(y) + 2 * lambd * coefficients), d.mean()]
            return value, gradient

        oracle = minimize(
            objective, np.zeros(len(X) + 1), jac=True, method="BFGS",
            options={"gtol": 1e-10, "maxiter": 2000},
        )
        self.assertTrue(np.isfinite(oracle.fun))
        self.assertLess(np.linalg.norm(objective(oracle.x)[1]), 1e-6, oracle.message)
        model = KernelSGDDWD(
            lambd=lambd, gamma=gamma, max_epochs=1800, random_state=12
        ).fit(X, y)
        fitted = np.r_[model.dual_coef_.ravel(), model.intercept_[0]]
        gap = objective(fitted)[0] - oracle.fun
        self.assertGreaterEqual(gap, -1e-8)
        self.assertLess(gap, 0.04, f"finite-budget RKHS objective gap={gap}")
        assert_allclose(model.predict(X), y)

    def test_imbalanced_duplicate_kernel_rows_approach_free_intercept_oracle(self):
        # Conflicting labels at x=-1 make perfect classification impossible.
        # Repeated rows also make the Gram matrix singular; no inverse is used.
        X = np.array([[-2.0], [-1.0], [-1.0], [-0.3], [0.3],
                      [1.0], [1.0], [2.0], [2.0]])
        y = np.array([-1, -1, 1, -1, -1, 1, 1, -1, -1])
        gamma, lambd = 0.7, 0.04
        K = independent_rbf(X, X, gamma)
        self.assertEqual(np.count_nonzero(y == -1), 6)
        self.assertEqual(np.count_nonzero(y == 1), 3)
        self.assertLess(np.linalg.matrix_rank(K), len(X))

        def objective(parameters):
            coefficients, intercept = parameters[:-1], parameters[-1]
            loss, derivative = independent_loss_and_derivative(y * (K @ coefficients + intercept))
            d = derivative * y
            value = loss.mean() + lambd * (coefficients @ K @ coefficients)
            gradient = np.r_[K @ (d / len(y) + 2 * lambd * coefficients), d.mean()]
            return value, gradient

        oracle = minimize(
            objective, np.zeros(len(X) + 1), jac=True, method="BFGS",
            options={"gtol": 1e-10, "maxiter": 2000},
        )
        self.assertTrue(np.isfinite(oracle.fun))
        self.assertLess(np.linalg.norm(objective(oracle.x)[1]), 1e-6, oracle.message)
        model = KernelSGDDWD(
            lambd=lambd, gamma=gamma, max_epochs=1800, random_state=12,
            batch_size=1, sampling="with_replacement", stopping="fixed",
            optimizer_backend="python", prediction_backend="direct",
        ).fit(X, y)
        fitted = np.r_[model.dual_coef_.ravel(), model.intercept_[0]]
        value = objective(fitted)[0]
        gap = value - oracle.fun
        self.assertGreaterEqual(gap, -1e-8)
        self.assertLess(gap, 0.04, f"finite-budget imbalanced RKHS objective gap={gap}")
        assert_allclose(model.objective(X, y), value, rtol=0, atol=1e-12)
        self.assertEqual(model.n_updates_, len(X) * 1800)
        self.assertTrue(model.fit_usable_)
        self.assertTrue(model.cv_eligible_)
        self.assertFalse(model.optimality_checked_)
        self.assertFalse(model.converged_)

    def test_wide_linear_data_approach_constrained_free_intercept_oracle(self):
        # Every feature varies: this is genuinely wide, not zero-padded data.
        X = np.random.default_rng(20261017).normal(size=(12, 128)) / np.sqrt(128.0)
        y = np.array([-1] * 6 + [1] * 6)
        self.assertTrue(np.all(np.ptp(X, axis=0) > 0))
        self.assertEqual(np.linalg.matrix_rank(X), len(X))

        def objective(parameters):
            weights, intercept = parameters[:-1], parameters[-1]
            loss, derivative = independent_loss_and_derivative(y * (X @ weights + intercept))
            d = derivative * y
            return loss.mean(), np.r_[X.T @ d / len(y), d.mean()]

        oracle = minimize(
            objective, np.zeros(X.shape[1] + 1), jac=True, method="SLSQP",
            constraints={
                "type": "ineq", "fun": lambda p: 1.0 - p[:-1] @ p[:-1],
                "jac": lambda p: np.r_[-2.0 * p[:-1], 0.0],
            },
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        self.assertTrue(oracle.success, oracle.message)
        self.assertTrue(np.isfinite(oracle.fun))
        self.assertLessEqual(np.linalg.norm(oracle.x[:-1]), 1.0 + 1e-10)
        gradient = objective(oracle.x)[1]
        proposal = oracle.x[:-1] - gradient[:-1]
        projection = proposal / max(1.0, float(np.linalg.norm(proposal)))
        residual = np.linalg.norm(np.r_[oracle.x[:-1] - projection, gradient[-1]])
        self.assertLess(residual, 1e-6, oracle.message)
        model = LinearSGDDWD(
            max_epochs=600, random_state=12, batch_size=1,
            sampling="with_replacement", stopping="fixed", optimizer_backend="python",
        ).fit(X, y)
        fitted = np.r_[model.coef_.ravel(), model.intercept_[0]]
        value = objective(fitted)[0]
        gap = value - oracle.fun
        self.assertGreaterEqual(gap, -1e-8)
        self.assertLess(gap, 0.025, f"finite-budget wide linear objective gap={gap}")
        self.assertLessEqual(np.linalg.norm(fitted[:-1]), 1.0 + 1e-14)
        assert_allclose(model.objective(X, y), value, rtol=0, atol=1e-12)
        self.assertEqual(model.n_updates_, len(X) * 600)
        self.assertTrue(model.fit_usable_)
        self.assertTrue(model.cv_eligible_)
        self.assertFalse(model.optimality_checked_)
        self.assertFalse(model.converged_)

    def test_subset_kernel_reuses_actual_subset_mean_without_extra_weights(self):
        X = np.column_stack((np.linspace(-2, 2, 40), np.cos(np.arange(40)) * 0.03))
        y = np.where(X[:, 0] < 0, -1, 1)
        two_stage = TwoStageDWDClassifier(
            max_epochs=5, random_state=23, retention_fraction=0.4,
            gamma=0.6, lambd=0.015,
        ).fit(X, y)
        chosen = two_stage.selected_indices_
        direct = KernelSGDDWD(
            max_epochs=5, gamma=0.6, lambd=0.015,
            random_state=two_stage.stage_seeds_["kernel_sgd"],
        ).fit(X[chosen], y[chosen])
        # Identical sampled-index stream and actual subset imply identical
        # fitted states. This would fail if discarded rows changed a gradient
        # divisor, a regularizer, or an importance weight in the pipeline.
        assert_allclose(two_stage.kernel_estimator_.dual_coef_, direct.dual_coef_, rtol=0, atol=0)
        assert_allclose(two_stage.kernel_estimator_.intercept_, direct.intercept_, rtol=0, atol=0)
        assert_allclose(two_stage.decision_function(X), direct.decision_function(X), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
