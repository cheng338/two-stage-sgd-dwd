"""Numerical checks of the prescribed gradients, geometry, and selection."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from two_stage_sgd_dwd._core import (
    dwd_loss,
    dwd_loss_derivative,
    kernel_step,
    linear_step,
    project_unit_ball,
    rbf_kernel,
    select_near_boundary,
)


def finite_difference(function, value, epsilon=1e-6):
    value = np.asarray(value, dtype=float)
    result = np.empty_like(value)
    for index in np.ndindex(value.shape):
        change = np.zeros_like(value)
        change[index] = epsilon
        result[index] = (function(value + change) - function(value - change)) / (2 * epsilon)
    return result


def test_loss_branches_boundary_and_large_margin_are_safe():
    margins = np.array([-2.0, 0.0, 0.5, 1.0, 2.0, 1e308])
    original = margins.copy()
    with np.errstate(all="raise"):
        losses = dwd_loss(margins)
        derivatives = dwd_loss_derivative(margins)
    assert_allclose(losses[:5], [3, 1, 0.5, 0.25, 0.125])
    assert_allclose(derivatives[:5], [-1, -1, -1, -0.25, -0.0625])
    assert np.isfinite(losses[-1]) and losses[-1] > 0
    assert derivatives[-1] == 0.0  # Ordinary float64 underflow is permitted.
    assert isinstance(dwd_loss(0.0), float)
    assert isinstance(dwd_loss_derivative(0.0), float)
    assert_array_equal(margins, original)
    assert_allclose(dwd_loss(np.array([[0.0], [1.0]])), [[1.0], [0.25]])
    assert_allclose(dwd_loss_derivative(0.5 - 1e-9), dwd_loss_derivative(0.5 + 1e-9), atol=1e-8)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
def test_loss_derivative_matches_finite_differences(epsilon):
    margins = np.array([-1.2, 0.1, 0.8, 2.5])
    actual = finite_difference(lambda u: np.sum(dwd_loss(u)), margins, epsilon)
    assert_allclose(actual, dwd_loss_derivative(margins), atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_losses_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        dwd_loss(value)
    with pytest.raises(ValueError, match="finite"):
        dwd_loss_derivative([0.0, value])


@pytest.mark.parametrize("vector,expected", [([0, 0], [0, 0]), ([0.2, -0.3], [0.2, -0.3]), ([3, 4], [0.6, 0.8])])
def test_projection_inside_outside_and_zero_does_not_alias(vector, expected):
    original = np.asarray(vector, dtype=float)
    result = project_unit_ball(original)
    assert_allclose(result, expected)
    assert not np.shares_memory(result, original)
    assert np.linalg.norm(result) <= 1.0 + 1e-15


def test_projection_avoids_norm_overflow():
    projected = project_unit_ball([1e308, -1e308])
    assert_allclose(projected, [1 / np.sqrt(2), -1 / np.sqrt(2)])


def test_linear_one_step_projects_slope_only_and_uses_old_state():
    w = np.array([0.3, -0.4])
    x = np.array([3.0, 2.0])
    old_w, old_x = w.copy(), x.copy()
    b, y, eta = 1.1, -1, 0.7
    q = -1.0 * y  # Margin is negative, so the loss derivative is -1.
    expected_w = project_unit_ball(w - eta * q * x)
    actual_w, actual_b = linear_step(w, b, x, y, eta)
    assert_allclose(actual_w, expected_w)
    assert actual_b == pytest.approx(b - eta * q)
    assert isinstance(actual_b, float)
    assert_array_equal(w, old_w)
    assert_array_equal(x, old_x)
    # A nonzero intercept is unchanged when eta=0, even if slope needs projection.
    _, preserved_b = linear_step([3.0, 4.0], 2.7, [0.0, 0.0], 1, 0)
    assert preserved_b == 2.7


def test_linear_unbiased_gradient_matches_average_objective():
    X = np.array([[0.2, 0.8], [-0.7, 0.3], [1.1, -0.2]])
    y = np.array([1.0, -1.0, 1.0])
    theta = np.array([0.12, -0.18, 0.7])
    eta = 1e-3  # No projection activates at this interior point.
    gradients = []
    for x_i, y_i in zip(X, y):
        w, b = linear_step(theta[:-1], theta[-1], x_i, y_i, eta)
        gradients.append((theta - np.r_[w, b]) / eta)
    numerical = finite_difference(lambda t: np.mean(dwd_loss(y * (X @ t[:-1] + t[-1]))), theta)
    assert_allclose(np.mean(gradients, axis=0), numerical, atol=1e-7, rtol=1e-5)


def test_kernel_one_step_matches_function_update_and_preserves_inputs():
    X = np.array([[0.0], [0.7], [1.4]])
    probes = np.array([[-0.3], [0.4], [1.2]])
    K = rbf_kernel(X, X, 0.8)
    K_probe = rbf_kernel(X, probes, 0.8)
    a = np.array([0.2, -0.3, 0.15])
    old_a, column = a.copy(), K[:, 1].copy()
    old_column = column.copy()
    b, eta, lambd, i, y_i = 0.7, 0.08, 0.1, 1, 1
    q = float(dwd_loss_derivative(y_i * (a @ column + b))) * y_i
    updated, intercept = kernel_step(a, b, column, i, y_i, eta, lambd=lambd)
    expected_function = (1 - 2 * eta * lambd) * (a @ K_probe) - eta * q * K_probe[i]
    assert_allclose(updated @ K_probe, expected_function, atol=1e-14)
    assert intercept == pytest.approx(b - eta * q)
    assert isinstance(intercept, float)
    assert_array_equal(a, old_a)
    assert_array_equal(column, old_column)
    assert not np.shares_memory(updated, a)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
def test_kernel_unbiased_rkhs_gradient_and_coefficient_geometry(epsilon):
    X = np.array([[0.0], [0.6], [1.2]])
    K = rbf_kernel(X, X, 0.8)
    assert np.linalg.eigvalsh(K).min() > 0
    assert not np.allclose(K, np.eye(3))
    a = np.array([0.4, -0.2, 0.3])
    b, lambd, eta = 0.37, 0.1, 1e-3
    y = np.array([1, -1, 1])
    sampled_r, sampled_b = [], []
    for i in range(len(y)):
        updated, intercept = kernel_step(a, b, K[:, i], i, y[i], eta, lambd)
        sampled_r.append((a - updated) / eta)
        sampled_b.append((b - intercept) / eta)
    d = dwd_loss_derivative(y * (K @ a + b)) * y
    r = 2 * lambd * a + d / len(y)
    assert_allclose(np.mean(sampled_r, axis=0), r, atol=1e-12)
    assert_allclose(np.mean(sampled_b), np.mean(d), atol=1e-12)

    def objective(coef):
        return np.mean(dwd_loss(y * (K @ coef + b))) + lambd * (coef @ K @ coef)

    numerical = finite_difference(objective, a, epsilon)
    assert_allclose(numerical, K @ r, atol=1e-7, rtol=1e-5)
    assert not np.allclose(numerical, r, atol=1e-4)  # Euclidean != RKHS coefficients.


def test_rbf_parameterization_symmetry_and_large_common_offset():
    X = np.array([[0, 1], [2, 3]], dtype=float)
    Y = np.array([[1, -1], [4, 2], [0, 1]], dtype=float)
    original_X, original_Y = X.copy(), Y.copy()
    expected = np.exp(-0.3 * np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    assert_allclose(rbf_kernel(X, Y, 0.3), expected)
    assert_allclose(rbf_kernel(X, X, 0.3).diagonal(), 1.0)
    assert_allclose(rbf_kernel(X, Y, 0.3), rbf_kernel(Y, X, 0.3).T)
    assert_array_equal(X, original_X)
    assert_array_equal(Y, original_Y)
    shifted = np.array([[1e12, 1e12], [1e12 + 1, 1e12 + 2]])
    assert_allclose(rbf_kernel(shifted, shifted, 0.3)[0, 1], np.exp(-1.5))
    assert rbf_kernel(np.empty((0, 2)), Y, 0.3).shape == (0, 3)


def test_selection_is_global_unique_nonmutating_and_repairs_legacy_counterexample():
    scores = np.array([-0.9, 0.01, -0.2, 0.03, -0.4, 0.05])
    original = scores.copy()
    selected = select_near_boundary(scores, 0.5, np.random.default_rng(7))
    assert_array_equal(selected, [1, 3, 5])
    assert np.unique(selected).size == len(selected)
    assert_array_equal(scores, original)
    assert len(select_near_boundary(scores, 0.01, np.random.default_rng(7))) == 1
    assert len(select_near_boundary(scores, 1, np.random.default_rng(7))) == len(scores)


def test_selection_tie_keys_are_reproducible_label_independent_and_use_ids():
    scores = np.array([-0.1, 0.1, -0.1, 0.1, 3.0])
    ids = np.array(["row-z", "row-a", "row-c", "row-b", "row-q"])
    seed = 394
    expected = np.lexsort((ids, np.random.default_rng(seed).random(len(scores)), np.abs(scores)))[:3]
    selected = select_near_boundary(scores, 0.5, np.random.default_rng(seed), ids)
    assert_array_equal(selected, expected)
    assert_array_equal(selected, select_near_boundary(-scores, 0.5, np.random.default_rng(seed), ids))
    assert_array_equal(ids, ["row-z", "row-a", "row-c", "row-b", "row-q"])
    assert select_near_boundary([], 0.05, np.random.default_rng(seed)).shape == (0,)


def test_selection_uses_original_ids_when_random_keys_also_tie():
    class TiedKeyGenerator(np.random.Generator):
        def random(self, size=None):
            return np.zeros(size)

    generator = TiedKeyGenerator(np.random.PCG64(7))
    selected = select_near_boundary([0.0, 0.0, 0.0], 0.5, generator, [9, 2, 5])
    assert_array_equal(selected, [1, 2])


@pytest.mark.parametrize("fraction", [0, -0.1, 1.01, np.nan, np.inf])
def test_selection_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        select_near_boundary([1.0], fraction, np.random.default_rng(4))


def test_core_validates_shapes_labels_indices_and_nonfinite_calculations():
    with pytest.raises(ValueError):
        dwd_loss(np.array([1.0 + 2.0j]))
    with pytest.raises(ValueError):
        linear_step([0, 0], 0, [1], 1, 0.1)
    with pytest.raises(ValueError):
        linear_step([0], 0, [1], 0, 0.1)
    with pytest.raises(ValueError):
        linear_step([0], [0], [1], 1, 0.1)
    with pytest.raises(ValueError):
        kernel_step([0], 0, [1], 1, 1, 0.1, 0.2)
    with pytest.raises(ValueError):
        kernel_step([0], 0, [1], 0, 1, -0.1, 0.2)
    with pytest.raises(ValueError):
        kernel_step([0], 0, [1], 0, 1, 0.1, 0)
    with pytest.raises(ValueError):
        rbf_kernel([[1, 2]], [[1]], 1)
    with pytest.raises(ValueError):
        rbf_kernel([[1]], [[1]], np.nan)
    with pytest.raises(ValueError):
        select_near_boundary([0, 0], 0.5, np.random.default_rng(1), [1, 1])
    with pytest.raises(FloatingPointError):
        linear_step([1e308], 0, [1e308], 1, 0.1)
    with pytest.raises(FloatingPointError):
        rbf_kernel([[1e308]], [[-1e308]], 1)
