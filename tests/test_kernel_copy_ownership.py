"""Kernel blocks remain independently owned when producers retain their arrays."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from two_stage_sgd_dwd import _kernels
from two_stage_sgd_dwd._kernels import resolve_kernel


class ArraySubclass(np.ndarray):
    pass


def retained_array(kind):
    values = np.arange(6, dtype=np.float64).reshape(3, 2).copy()
    if kind == "readonly":
        values.flags.writeable = False
    elif kind == "view":
        values = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    elif kind == "subclass":
        derived = np.ndarray.__new__(ArraySubclass, values.shape, dtype=np.float64)
        derived[:] = values
        values = derived
    elif kind == "float32":
        values = values.astype(np.float32)
    elif kind == "fortran":
        values = np.asfortranarray(values)
    return values


@pytest.mark.parametrize("producer", ["callable", "pairwise", "rbf"])
@pytest.mark.parametrize("kind", ["owned", "readonly", "view", "subclass", "float32", "fortran"])
def test_external_producer_cannot_retain_alias_to_returned_block(monkeypatch, producer, kind):
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    Y = X[:2].copy()
    retained = retained_array(kind)
    expected = np.array(retained, dtype=np.float64, copy=True)

    def kernel(*args, **kwargs):
        return retained

    if producer == "callable":
        spec = resolve_kernel(X, kernel)
    elif producer == "pairwise":
        monkeypatch.setattr(_kernels, "pairwise_kernels", kernel)
        spec = resolve_kernel(X, "linear")
    else:
        monkeypatch.setattr(_kernels, "rbf_kernel", kernel)
        spec = resolve_kernel(X, "rbf", gamma=0.5)

    result = spec.evaluate(X, Y)
    assert type(result) is np.ndarray
    assert result.dtype == np.float64
    assert result.flags.owndata and result.flags.writeable
    assert not np.shares_memory(result, retained)
    assert_array_equal(result, expected)
    result[0, 0] = -100.0
    assert_array_equal(retained, expected)
    retained.flags.writeable = True
    retained[:] = -200.0
    expected[0, 0] = -100.0
    assert_array_equal(result, expected)


@pytest.mark.parametrize("name,gamma,kws", [
    ("cosine", 1.0, None),
    ("rbf", 0.0, None),
    ("poly", 1e308, {"degree": 0, "coef0": 0.0}),
    ("polynomial", 1.0, {"degree": 0}),
    ("poly", 0.0, {"degree": 3, "coef0": 2.0}),
    ("polynomial", 0.0, {"degree": 2, "coef0": 0.5}),
    ("laplacian", 0.0, None),
    ("chi2", 0.0, None),
    ("sigmoid", 0.0, {"coef0": -0.5}),
])
@pytest.mark.parametrize("magnitude", [1.0, 1e-300, 1e300])
def test_local_blocks_are_private_across_inputs_and_calls(name, gamma, kws, magnitude):
    X = magnitude * np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    Y = X[:2].copy()
    original_X, original_Y = X.copy(), Y.copy()
    spec = resolve_kernel(X, name, gamma=gamma, kernel_kws=kws)
    first = spec.evaluate(X, Y)
    expected = first.copy()
    first[:] = -100.0
    second = spec.evaluate(X, Y)
    assert type(second) is np.ndarray
    assert second.dtype == np.float64
    assert second.flags.owndata and second.flags.writeable
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(second, X)
    assert not np.shares_memory(second, Y)
    assert_array_equal(second, expected)
    assert_array_equal(X, original_X)
    assert_array_equal(Y, original_Y)


def test_cosine_subclass_hook_cannot_retain_alias_to_returned_block(monkeypatch):
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    retained = np.arange(6, dtype=np.float64).reshape(3, 2).copy()
    expected = retained.copy()

    class MatmulSubclass(np.ndarray):
        def __matmul__(self, other):
            return retained

    monkeypatch.setattr(_kernels, "_unit_rows", lambda rows: rows.view(MatmulSubclass))
    result = resolve_kernel(X, "cosine").evaluate(X, X[:2])
    assert type(result) is np.ndarray
    assert not np.shares_memory(result, retained)
    retained[:] = -100.0
    assert_array_equal(result, expected)
    result[:] = -200.0
    assert_array_equal(retained, np.full((3, 2), -100.0))
