"""Real-value contracts include represented object contents and conversion hooks."""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from two_stage_sgd_dwd import KernelSGDDWD, optimization_diagnostics
from two_stage_sgd_dwd import _core, diagnostics
from two_stage_sgd_dwd._kernels import KernelSpec


def object_array(item):
    value = np.empty(1, dtype=object)
    value[0] = item
    return value


@pytest.mark.parametrize("item", [complex(1, 0), complex(1, 2), np.complex64(1),
                                np.complex128(1 + 2j), np.array(1 + 0j),
                                np.array([np.complex128(1)], dtype=object)])
def test_complex_object_contents_rejected_before_real_conversion(item):
    value = object_array(item)
    assert _core._contains_complex(value)
    with pytest.raises(ValueError, match="real"):
        _core._finite_array(value, "values")
    with pytest.raises(ValueError, match="real"):
        diagnostics._array(value, "values", 1)


def test_cyclic_nested_object_walk_terminates_and_finds_complex():
    first, second = np.empty(2, dtype=object), np.empty(2, dtype=object)
    first[0], first[1] = second, 1.0
    second[0], second[1] = first, 2.0
    assert not _core._contains_complex(first)
    second[1] = np.complex128(2)
    assert _core._contains_complex(first)
    with pytest.raises(ValueError, match="real"):
        _core._finite_array(first, "values")


def test_real_objects_use_original_conversion_without_class_property_hooks():
    events = []

    class Real:
        @property
        def __class__(self):
            raise AssertionError("The represented type must be inspected directly")

        def __float__(self):
            events.append("float")
            return 0.25

    value = np.array([1.0, 2, True, Decimal("0.5"), Fraction(3, 4), Real()], dtype=object)
    assert not _core._contains_complex(value)
    assert events == []
    np.testing.assert_array_equal(_core._finite_array(value, "values"),
                                  [1.0, 2.0, 1.0, 0.5, 0.75, 0.25])
    assert events == ["float"]


@pytest.mark.parametrize("kind", ["complex", "nested_complex", "real"])
def test_callable_kernel_contents_are_checked_and_output_ownership_is_preserved(kind):
    matrix = np.ones((2, 2), dtype=object)
    if kind == "complex":
        matrix[0, 1] = np.complex128(1)
    elif kind == "nested_complex":
        matrix[0, 1] = np.array(np.complex64(1))
    else:
        matrix[0, 1] = Decimal("1")
    supplied = matrix.copy()
    spec = KernelSpec(lambda X, Y: matrix, None, {})
    X = np.array([[0.0], [1.0]])
    if kind != "real":
        with pytest.raises(ValueError, match="real"):
            spec.evaluate(X, X)
    else:
        result = spec.evaluate(X, X)
        np.testing.assert_array_equal(result, np.ones((2, 2)))
        result[0, 0] = 8.0
    assert matrix[0, 0] == supplied[0, 0] == 1
    assert matrix[0, 1] is supplied[0, 1]


def test_diagnostics_reject_hidden_complex_fitted_coefficients():
    X = np.array([[-1.0], [1.0]])
    y = np.array([-1, 1])
    model = KernelSGDDWD(max_epochs=1, stopping="fixed", optimizer_backend="python",
                         random_state=7).fit(X, y)
    model.dual_coef_ = model.dual_coef_.astype(object)
    model.dual_coef_[0, 0] = np.complex128(model.dual_coef_[0, 0])
    with pytest.raises(ValueError, match="real"):
        optimization_diagnostics(model, X, y)


def test_conversion_failure_identity_and_finite_checks_remain():
    marker = RuntimeError("conversion failed")

    class Broken:
        def __float__(self):
            raise marker

    with pytest.raises(RuntimeError) as caught:
        _core._finite_array(object_array(Broken()), "values")
    assert caught.value is marker
    for item in (float("inf"), float("nan"), Decimal("Infinity")):
        with pytest.raises(ValueError, match="finite"):
            _core._finite_array(object_array(item), "values")
