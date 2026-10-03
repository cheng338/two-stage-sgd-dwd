"""Independent mathematical checks for optional averaged batch updates."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from two_stage_sgd_dwd._core import (
    dwd_loss,
    dwd_loss_derivative,
    kernel_batch_step,
    kernel_step,
    linear_batch_step,
    linear_step,
    rbf_kernel,
)


def finite_difference(function, value, epsilon):
    value = np.asarray(value, dtype=float)
    gradient = np.empty_like(value)
    for index in np.ndindex(value.shape):
        perturbation = np.zeros_like(value)
        perturbation[index] = epsilon
        gradient[index] = (
            function(value + perturbation) - function(value - perturbation)
        ) / (2 * epsilon)
    return gradient


def test_batch_size_one_matches_existing_linear_and_kernel_steps():
    w = np.array([0.2, -0.3])
    x = np.array([1.2, 0.8])
    b, y, eta = 0.65, 1.0, 0.03
    scalar_w, scalar_b = linear_step(w, b, x, y, eta)
    batch_w, batch_b = linear_batch_step(w, b, x[None, :], [y], eta)
    assert_allclose(batch_w, scalar_w, atol=1e-14)
    assert_allclose(batch_b, scalar_b, atol=1e-14)
    a, k = np.array([0.3, -0.2, 0.4]), np.array([0.4, 1.0, 0.8])
    scalar_a, scalar_b = kernel_step(a, b, k, 1, y, eta, 0.1)
    batch_a, batch_b = kernel_batch_step(a, b, [a @ k + b], [1], [y], eta, 0.1)
    assert_allclose(batch_a, scalar_a, atol=1e-14)
    assert_allclose(batch_b, scalar_b, atol=1e-14)
    assert isinstance(batch_b, float)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
def test_linear_full_batch_gradient_matches_independent_finite_difference(epsilon):
    X = np.array([[0.4, 0.8], [-0.7, 0.2], [1.1, -0.4]])
    labels = np.array([1.0, -1.0, 1.0])
    theta = np.array([0.1, -0.15, 0.8])
    eta = 1e-3  # Remain inside the ball so projection does not alter the gradient.
    updated_w, updated_b = linear_batch_step(theta[:-1], theta[-1], X, labels, eta)
    gradient = (theta - np.r_[updated_w, updated_b]) / eta
    numerical = finite_difference(
        lambda state: np.mean(dwd_loss(labels * (X @ state[:-1] + state[-1]))),
        theta,
        epsilon,
    )
    assert_allclose(gradient, numerical, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
def test_kernel_full_batch_has_rkhs_not_euclidean_coefficient_gradient(epsilon):
    X = np.array([[0.0], [0.6], [1.4]])
    K = rbf_kernel(X, X, 0.7)
    assert np.linalg.eigvalsh(K).min() > 0
    assert not np.allclose(K, np.eye(len(K)))
    a, b = np.array([0.4, -0.2, 0.3]), 0.8
    labels = np.array([1.0, -1.0, 1.0])
    eta, lambd = 1e-3, 0.1
    updated_a, updated_b = kernel_batch_step(
        a, b, K @ a + b, np.arange(len(a)), labels, eta, lambd=lambd
    )
    rkhs_coefficients = (a - updated_a) / eta
    numerical = finite_difference(
        lambda value: np.mean(dwd_loss(labels * (K @ value + b)))
        + lambd * (value @ K @ value),
        a,
        epsilon,
    )
    assert_allclose(numerical, K @ rkhs_coefficients, atol=1e-7, rtol=1e-5)
    assert not np.allclose(numerical, rkhs_coefficients, atol=1e-4)
    numerical_b = finite_difference(
        lambda value: np.mean(dwd_loss(labels * (K @ a + value[0]))),
        [b],
        epsilon,
    )[0]
    assert_allclose((b - updated_b) / eta, numerical_b, atol=1e-7, rtol=1e-5)


def test_kernel_repeated_indices_accumulate_and_shrink_only_once():
    a, b = np.array([0.3, -0.2, 0.1]), 0.6
    indices = np.array([0, 0, 2])
    labels = np.array([1.0, 1.0, -1.0])
    scores = np.array([0.4, 0.4, -0.7])
    eta, lambd = 0.12, 0.15
    derivative = np.array([-1.0, -1.0, -0.25 / 0.7**2])
    q = derivative * labels
    expected = (1 - 2 * eta * lambd) * a
    expected[0] -= eta * (q[0] + q[1]) / 3
    expected[2] -= eta * q[2] / 3
    updated, updated_b = kernel_batch_step(a, b, scores, indices, labels, eta, lambd)
    assert_allclose(updated, expected, atol=1e-14)
    assert updated_b == pytest.approx(b - eta * q.mean())
    # Duplicating a complete batch leaves the mean gradient unchanged.
    doubled_a, doubled_b = kernel_batch_step(
        a, b, np.tile(scores, 2), np.tile(indices, 2), np.tile(labels, 2), eta, lambd
    )
    assert_allclose(doubled_a, updated, atol=1e-14)
    assert_allclose(doubled_b, updated_b, atol=1e-14)


def test_linear_batch_averages_then_projects_once_without_shrinking_intercept():
    w, b = np.array([0.6, 0.8]), 2.5
    X = np.array([[3.0, 4.0], [3.0, 4.0]])
    labels = np.array([-1.0, -1.0])
    updated, updated_b = linear_batch_step(w, b, X, labels, 0.8)
    assert_allclose(updated, [-0.6, -0.8])
    assert updated_b == pytest.approx(1.7)
    doubled_w, doubled_b = linear_batch_step(w, b, np.tile(X, (2, 1)), np.tile(labels, 2), 0.8)
    assert_allclose(doubled_w, updated)
    assert_allclose(doubled_b, updated_b)


def test_batch_margins_use_old_complete_state_and_never_mutate_inputs():
    w, X, y = np.array([0.2]), np.array([[2.0], [-1.0]]), np.array([1.0, -1.0])
    b, eta = 0.3, 0.2
    originals = [value.copy() for value in (w, X, y)]
    q = dwd_loss_derivative(y * (X @ w + b)) * y
    updated, updated_b = linear_batch_step(w, b, X, y, eta)
    assert_allclose(updated, w - eta * (q @ X) / 2)
    assert_allclose(updated_b, b - eta * q.mean())
    for supplied, original in zip((w, X, y), originals):
        assert_array_equal(supplied, original)
    assert not np.shares_memory(w, updated)
    assert isinstance(updated_b, float)

    a, scores, indices = np.array([0.2, -0.1]), np.array([0.7, -0.1]), np.array([0, 1])
    originals = [value.copy() for value in (a, scores, indices, y)]
    q = dwd_loss_derivative(y * scores) * y
    updated, updated_b = kernel_batch_step(a, b, scores, indices, y, eta, 0.15)
    assert_allclose(updated, (1 - 2 * eta * 0.15) * a - eta * q / 2)
    assert_allclose(updated_b, b - eta * q.mean())
    for supplied, original in zip((a, scores, indices, y), originals):
        assert_array_equal(supplied, original)
    assert not np.shares_memory(a, updated)
    assert isinstance(updated_b, float)


@pytest.mark.parametrize(
    "X,y",
    [([], []), (np.empty((0, 1)), []), ([[1, 2]], [1]), ([[1]], []),
     ([[1]], [0]), ([[1]], [[1]]), ([[np.inf]], [1])],
)
def test_linear_batch_rejects_empty_or_malformed_data(X, y):
    with pytest.raises(ValueError):
        linear_batch_step([0.0], 0.0, X, y, 0.1)


@pytest.mark.parametrize(
    "scores,indices,y",
    [([], [], []), ([0], [0.0], [1]), ([0], [True], [1]), ([0], [-1], [1]),
     ([0], [2], [1]), ([0], [0, 1], [1]), ([0], [0], [0]),
     ([np.nan], [0], [1]), ([0], [[0]], [1])],
)
def test_kernel_batch_rejects_empty_or_malformed_data(scores, indices, y):
    with pytest.raises(ValueError):
        kernel_batch_step([0.0, 0.0], 0.0, scores, indices, y, 0.1, 0.2)


def test_batch_steps_reject_nonfinite_or_invalid_parameters_and_arithmetic():
    with pytest.raises(ValueError):
        linear_batch_step([0], np.inf, [[1]], [1], 0.1)
    with pytest.raises(ValueError):
        linear_batch_step([0], 0, [[1]], [1], -0.1)
    with pytest.raises(ValueError):
        kernel_batch_step([0], 0, [0], [0], [1], 0.1, 0)
    with pytest.raises(ValueError):
        kernel_batch_step([0], 0, [0], [0], [1], np.nan, 0.2)
    with pytest.raises(FloatingPointError):
        linear_batch_step([1e308], 0, [[1e308]], [1], 0.1)
    with pytest.raises(FloatingPointError):
        kernel_batch_step([1e308], 0, [0], [0], [1], 1e308, 1e308)
