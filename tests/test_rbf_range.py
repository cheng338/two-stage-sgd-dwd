"""Finite Gaussian range recovery and its original arithmetic/hook boundaries."""
from decimal import Decimal, localcontext
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from _rbf_range_oracles import scaled_squared_distance, zero_or_one_kernel
from two_stage_sgd_dwd import _core, _prediction, _training_kernels


@pytest.mark.parametrize("coordinate,gamma", [(2.0**520, 2.0**-1040),
                                              (2.0**540, 2.0**-1074),
                                              (2.0**-540, 2.0**1000)])
def test_overflow_or_underflow_distance_keeps_representable_exponent(coordinate, gamma):
    X, Y = np.array([[0.0]]), np.array([[coordinate]])
    exponent = scaled_squared_distance(X[0], Y[0], gamma)
    distances = cdist(X, Y, metric="sqeuclidean")
    _core._scale_rbf_distances(distances, X, Y, gamma, trusted=True)
    assert distances[0, 0] == -float(exponent)
    with localcontext() as context:
        context.prec = 180
        expected = float(-(Decimal(exponent.numerator) / Decimal(exponent.denominator)))
        expected = float(Decimal.from_float(expected).exp())
    actual = _core.rbf_kernel(X, Y, gamma)[0, 0]
    # These powers of two have exact representable exponents. This tolerance
    # is the previously qualified binary64 exponential budget, not an SGD gate.
    assert actual == pytest.approx(expected, rel=4 * np.finfo(float).eps, abs=0.0)


@pytest.mark.parametrize("gamma", [2.0**-1074, 0.2, np.finfo(float).max])
def test_identical_extreme_rows_are_one_including_empty_query(gamma):
    X = np.full((3, 2), 1e308)
    before = X.tobytes()
    np.testing.assert_array_equal(_core.rbf_kernel(X, X, gamma), np.ones((3, 3)))
    assert _core.rbf_kernel(X[:0], X, gamma).shape == (0, 3)
    assert X.tobytes() == before


def test_core_named_kernel_and_column_loader_share_range_recovery():
    from two_stage_sgd_dwd._kernels import KernelSpec
    X = np.array([[1e308], [-1e308]])
    gamma = 0.2
    expected = zero_or_one_kernel(X, X, gamma)
    np.testing.assert_array_equal(_core.rbf_kernel(X.astype(object), X, gamma), expected)
    np.testing.assert_array_equal(KernelSpec("rbf", gamma, {}).evaluate(X, X), expected)
    loader = _training_kernels.make_rbf_column_loader(X, gamma)
    for index in range(len(X)):
        np.testing.assert_array_equal(loader(index), expected[:, index])


def test_positive_tiny_contrast_is_recovered_but_rounded_kernel_stays_one():
    X = np.array([[-5e-163], [5e-163]])
    gamma = 1e24
    raw = cdist(X, X, metric="sqeuclidean")
    np.testing.assert_array_equal(raw, np.zeros_like(raw))
    exact = scaled_squared_distance(X[0], X[1], gamma)
    assert exact > 0 and float(exact) > 0
    _core._scale_rbf_distances(raw, X, X, gamma, trusted=True, contrast=True)
    assert raw[0, 1] == raw[1, 0] == -float(exact)
    np.testing.assert_array_equal(_core.rbf_kernel(X, X, gamma), np.ones((2, 2)))
    assert np.expm1(raw)[0, 1] < 0.0


def test_contrast_below_binary64_exponent_range_is_not_rescaled():
    X = np.array([[-2.0**-560], [2.0**-560]])
    exact = scaled_squared_distance(X[0], X[1], 1.0)
    assert exact > Fraction(0) and float(exact) == 0.0
    distances = cdist(X, X, metric="sqeuclidean")
    _core._scale_rbf_distances(distances, X, X, 1.0, trusted=True, contrast=True)
    np.testing.assert_array_equal(distances, np.zeros_like(distances))


def test_zero_distance_certificate_rejects_mixed_tiny_coordinates():
    threshold = 2.0**-458
    assert _core._rbf_zero_distances_are_exact(np.array([[0.0, threshold], [-threshold, 0.0]]))
    assert not _core._rbf_zero_distances_are_exact(np.array([[np.nextafter(threshold, 0.0)]]))
    mixed = np.array([[1.0, -5e-163], [1.0, 5e-163]])
    assert not _core._rbf_zero_distances_are_exact(mixed)
    distances = cdist(mixed, mixed, metric="sqeuclidean")
    _core._scale_rbf_distances(distances, mixed, mixed, 1e24, trusted=True, contrast=True)
    assert distances[0, 1] == -float(scaled_squared_distance(mixed[0], mixed[1], 1e24))


def test_duplicates_do_not_require_exact_distance_allocation():
    X = np.full((8, 3), 0.25)
    distances = cdist(X, X, metric="sqeuclidean")
    with patch("fractions.Fraction", side_effect=AssertionError("unneeded exact distance")):
        _core._scale_rbf_distances(distances, X, X, 0.2, trusted=True, contrast=True,
                                  zero_distances_exact=_core._rbf_zero_distances_are_exact(X))
    np.testing.assert_array_equal(distances, np.zeros((8, 8)))


def test_recovery_declines_unqualified_arithmetic_and_untrusted_distance():
    X, Y = np.array([[1e308]]), np.array([[-1e308]])
    for eligible, trusted in [(False, True), (True, False)]:
        with patch.object(_core, "_rbf_recovery_arithmetic", return_value=eligible), np.errstate(all="raise"):
            policy = np.geterr().copy()
            with pytest.raises(FloatingPointError):
                _core._scale_rbf_distances(np.array([[np.inf]]), X, Y, 1.0, trusted=trusted)
            assert np.geterr() == policy
            if not eligible:
                assert not _core._rbf_zero_distances_are_exact(np.ones((2, 1)))


@pytest.mark.parametrize("site", ["direct", "range_wrapper", "near_constant"])
def test_original_finite_result_hook_exception_retains_identity(site):
    X, Y = np.array([[0.25], [0.5]]), np.array([[-1.0], [1.0]])
    coefficients = np.array([-1.0, 1.0])
    marker = FloatingPointError("original distance hook")
    original = _prediction._finite_result

    def observe(values, name):
        if name == "squared Euclidean distances":
            raise marker
        return original(values, name)

    with patch.object(_prediction, "_finite_result", observe):
        with pytest.raises(FloatingPointError) as caught:
            if site == "direct":
                _prediction.rbf_prediction_scores(X, Y, coefficients, 0.25, 1e-20, 2)
            elif site == "range_wrapper":
                _prediction._range_checked_distance_rows_to_kernel(np.ones((2, 2)), 1e-20, X, Y)
            else:
                _prediction._near_constant_scores(X, Y, coefficients, 0.0, 1e-20,
                                                  np.ones((2, 2)), np.zeros(2), 1.0, None, True)
    assert caught.value is marker


def test_original_two_argument_converter_hook_has_precedence():
    marker = RuntimeError("original converter hook")
    with patch.object(_prediction, "_distance_rows_to_kernel", side_effect=marker):
        with pytest.raises(RuntimeError) as caught:
            _prediction._range_checked_distance_rows_to_kernel(
                np.ones((2, 2)), 0.1, np.ones((2, 1)), np.ones((2, 1)))
    assert caught.value is marker
