"""Guarded norm/dot prediction against the unchanged direct implementation."""
import pickle
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist
from _rbf_range_oracles import zero_or_one_kernel

from two_stage_sgd_dwd import KernelSGDDWD
from two_stage_sgd_dwd import _prediction


def inputs(n_basis=13, n_features=7, n_query=11):
    rng = np.random.default_rng(734)
    return (rng.normal(size=(n_query, n_features)),
            rng.normal(size=(n_basis, n_features)), rng.normal(size=n_basis))


def direct(X, basis, coefficients, intercept=0.13, gamma=0.2, batch=4):
    return _prediction.rbf_prediction_scores(X, basis, coefficients, intercept, gamma, batch)


def fast(X, basis, coefficients, intercept=0.13, gamma=0.2, batch=4):
    return _prediction.rbf_prediction_scores_blas(X, basis, coefficients, intercept, gamma, batch)


def bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('batch', [1, 4, 11, 100])
def test_well_separated_rows_use_blas_and_match_direct_within_roundoff(batch):
    X, basis, coefficients = inputs(n_features=33)
    expected = direct(X, basis, coefficients, batch=batch)
    with patch.object(_prediction, 'cdist', side_effect=AssertionError('Unneeded direct fallback')):
        actual = fast(X, basis, coefficients, batch=batch)
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
    assert_array_equal(actual >= 0, expected >= 0)


@pytest.mark.parametrize('offset', [1e12, -1e12, 1e150])
def test_large_common_offsets_fall_back_without_zero_clipping(offset):
    X, basis, coefficients = inputs()
    X, basis = X + offset, basis + offset
    expected = direct(X, basis, coefficients)
    calls = []

    def observed(left, right, **kwargs):
        calls.append(left.shape)
        return cdist(left, right, **kwargs)

    with patch.object(_prediction, 'cdist', side_effect=observed):
        actual = fast(X, basis, coefficients)
    bits(actual, expected)
    assert calls


def test_nearly_equal_nonidentical_rows_retain_their_positive_distance():
    basis = np.array([[1., 1., 1.], [2., 2., 2.], [-2., 1., 0.]])
    X = np.array([[1. + 2e-12, 1., 1.], [1., 1., 1.]])
    coefficients = np.array([1., 0., 0.])
    expected = direct(X, basis, coefficients, intercept=0., gamma=1e24)
    actual = fast(X, basis, coefficients, intercept=0., gamma=1e24)
    bits(actual, expected)
    assert 0.0 < actual[0] < 0.1
    assert actual[1] == 1.0


@pytest.mark.parametrize('value', [1e154, 1e308])
def test_overflowing_norms_with_finite_direct_distances_use_direct(value):
    basis = np.full((5, 3), value)
    X = np.full((7, 3), value)
    coefficients = np.linspace(-0.3, 0.7, len(basis))
    bits(fast(X, basis, coefficients), direct(X, basis, coefficients))


@pytest.mark.parametrize('kind', ['distance', 'gamma', 'coefficients', 'intercept'])
def test_finite_range_recovery_and_genuine_direct_failures(kind):
    X, basis, coefficients = inputs()
    gamma, intercept = 0.2, 0.13
    if kind == 'distance':
        X[:], basis[:] = 1e308, -1e308
    elif kind == 'gamma':
        gamma = 1e308
    elif kind == 'coefficients':
        coefficients[0] = np.nan
    else:
        intercept = np.inf
    for function in (direct, fast):
        if kind in ('distance', 'gamma'):
            kernel = zero_or_one_kernel(X, basis, gamma)
            assert_array_equal(kernel, np.zeros_like(kernel))
            actual = function(X, basis, coefficients, intercept=intercept, gamma=gamma)
            bits(actual, np.full(len(X), intercept))
        else:
            with pytest.raises(FloatingPointError):
                function(X, basis, coefficients, intercept=intercept, gamma=gamma)


@pytest.mark.parametrize('n_basis', [5, 13, 17])
@pytest.mark.parametrize('coefficient_scale', [1.0, 1e16])
def test_uncertain_score_recomputes_whole_original_block_exactly(n_basis, coefficient_scale):
    X, basis, coefficients = inputs(n_basis=n_basis)
    coefficients *= coefficient_scale
    # Make the first direct score zero without changing distances. Odd basis
    # counts exercise vector/tail layouts in the original full-block exp call.
    intercept = -direct(X, basis, coefficients, intercept=0., batch=4)[0]
    expected = direct(X, basis, coefficients, intercept=intercept, batch=4)
    calls = []

    def observed(left, right, **kwargs):
        calls.append(left.copy())
        return cdist(left, right, **kwargs)

    with patch.object(_prediction, 'cdist', side_effect=observed):
        actual = fast(X, basis, coefficients, intercept=intercept, batch=4)
    assert any(array.shape == X[:4].shape and np.array_equal(array, X[:4]) for array in calls)
    bits(actual[:4], expected[:4])
    assert_array_equal(actual >= 0, expected >= 0)
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-13 * coefficient_scale)


def test_extreme_finite_coefficient_mass_uses_direct_without_overflowing_guard():
    basis = np.zeros((2, 3))
    X = np.ones((3, 3))
    coefficients = np.array([1e308, -1e308])
    bits(fast(X, basis, coefficients), direct(X, basis, coefficients))


def test_subnormal_coefficient_score_guard_does_not_underflow_to_zero():
    X, basis, coefficients = inputs()
    coefficients *= 1e-320
    intercept = -direct(X, basis, coefficients, intercept=0., batch=4)[0]
    expected = direct(X, basis, coefficients, intercept=intercept, batch=4)
    calls = []

    def observed(left, right, **kwargs):
        calls.append(left.shape)
        return cdist(left, right, **kwargs)

    with patch.object(_prediction, 'cdist', side_effect=observed):
        actual = fast(X, basis, coefficients, intercept=intercept, batch=4)
    assert (4, X.shape[1]) in calls
    bits(actual[:4], expected[:4])
    assert_array_equal(actual >= 0, expected >= 0)


@pytest.mark.parametrize('gamma', [1e-300, 0.2, 1e300])
def test_extreme_widths_keep_finite_valid_scores(gamma):
    X, basis, coefficients = inputs()
    expected = direct(X, basis, coefficients, gamma=gamma)
    actual = fast(X, basis, coefficients, gamma=gamma)
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
    assert_array_equal(actual >= 0, expected >= 0)


def test_tiny_features_and_large_width_do_not_use_a_unit_distance_floor():
    X, basis, coefficients = inputs()
    X *= 1e-158
    basis *= 1e-158
    expected = direct(X, basis, coefficients, gamma=1e308)
    actual = fast(X, basis, coefficients, gamma=1e308)
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize('layout', ['fortran', 'strided', 'reversed'])
def test_layouts_readonly_and_inputs_are_unchanged(layout):
    X, basis, coefficients = inputs()
    if layout == 'fortran':
        X, basis = np.asfortranarray(X), np.asfortranarray(basis)
    elif layout == 'strided':
        X, basis = np.repeat(X, 2, axis=1)[:, ::2], np.repeat(basis, 2, axis=1)[:, ::2]
    else:
        X, basis = X[::-1], basis[::-1]
    before = [array.copy() for array in (X, basis, coefficients)]
    for array in (X, basis, coefficients):
        array.setflags(write=False)
    assert_allclose(fast(X, basis, coefficients), direct(X, basis, coefficients), rtol=2e-12, atol=2e-13)
    for original, array in zip(before, (X, basis, coefficients)):
        assert_array_equal(array, original)


def test_matmul_reuses_one_bounded_buffer_for_partial_last_block():
    X, basis, coefficients = inputs(n_features=33, n_query=11)
    original = np.matmul
    calls = []

    def observed(left, right, **kwargs):
        target = kwargs['out']
        calls.append((left.shape, target.shape, target.ctypes.data, target.base))
        assert target.flags.c_contiguous
        return original(left, right, **kwargs)

    with patch.object(_prediction.np, 'matmul', side_effect=observed):
        fast(X, basis, coefficients, batch=4)
    assert [item[0][0] for item in calls] == [4, 4, 3]
    assert len({item[2] for item in calls}) == 1
    assert all(item[3] is calls[0][3] for item in calls)
    assert calls[0][3].nbytes == 4 * len(basis) * 8


def test_fitted_state_and_serialized_predictions_remain_unchanged():
    X, _, _ = inputs(n_query=12)
    model = KernelSGDDWD(max_epochs=2, random_state=7,
                         prediction_backend='blas').fit(X, np.tile([-1, 1], 6))
    query, _, _ = inputs()
    before = pickle.dumps(model)
    first = model.decision_function(query)
    model.predict(query[::-1])
    assert pickle.dumps(model) == before
    bits(pickle.loads(before).decision_function(query), first)


def test_helper_validation_matches_direct_for_invalid_basis_width_and_shape():
    X, basis, coefficients = inputs()
    cases = [(basis * np.nan, coefficients, 0.2), (basis[:, :2], coefficients, 0.2),
             (basis, coefficients[:-1], 0.2), (basis, coefficients, np.inf)]
    for bad_basis, bad_coefficients, gamma in cases:
        for function in (direct, fast):
            with pytest.raises(ValueError):
                function(X, bad_basis, bad_coefficients, gamma=gamma)
