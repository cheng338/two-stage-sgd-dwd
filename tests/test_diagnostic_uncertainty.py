"""Reporting risk and the combined real-input/RBF-contrast diagnostic path."""
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest

from _rbf_range_oracles import scaled_squared_distance
from two_stage_sgd_dwd import KernelSGDDWD, optimization_diagnostics
from two_stage_sgd_dwd import _estimators, diagnostics


@pytest.mark.parametrize("dimensions", [1, 33])
@pytest.mark.parametrize("gamma", [1e24, 1e100, 1e250, 1e290, 1e292])
def test_actual_fit_tiny_contrast_oracle_and_risk_scale(dimensions, gamma):
    X = np.zeros((2, dimensions))
    X[:, 0] = [-5e-163, 5e-163]
    y = np.array([-1, 1])
    model = KernelSGDDWD(gamma=gamma, lambd=0.1, batch_size="full", max_epochs=1,
                         stopping="fixed", optimizer_backend="python", random_state=7).fit(X, y)
    with localcontext() as context:
        context.prec = 500
        exponent = scaled_squared_distance(X[0], X[1], gamma)
        contrast = (-(Decimal(exponent.numerator) / Decimal(exponent.denominator))).exp() - 1
        a = [Decimal.from_float(float(value)) for value in model.dual_coef_[0]]
        b = Decimal.from_float(float(model.intercept_[0]))
        penalty = Decimal.from_float(float(model.lambd_))
        constant = sum(a) + b
        expected_scores = np.array([float(constant + contrast * a[1]),
                                    float(constant + contrast * a[0])])
        expected_norm = float(sum(a) ** 2 + 2 * contrast * a[0] * a[1])
        t = [2 * penalty * value - Decimal(int(label)) / 2 for value, label in zip(a, y)]
        expected_gradient = float((sum(t) ** 2 + 2 * contrast * t[0] * t[1]).sqrt())
        expected_magnitude = float(abs(contrast))
    # The existing direct backend intentionally uses rounded Gaussian entries;
    # the analytic BLAS/validation/diagnostic consumers use expm1 contrasts.
    model.prediction_backend = "direct"
    np.testing.assert_array_equal(model.decision_function(X), np.zeros(2))
    model.prediction_backend = "blas"
    np.testing.assert_allclose(model.decision_function(X), expected_scores,
                               rtol=16 * np.finfo(float).eps, atol=0.0)
    validation = _estimators._validation_scores(
        "kernel", X, model.dual_coef_[0], float(model.intercept_[0]), X,
        model.gamma_, 2, kernel_spec=model._kernel_spec_)
    np.testing.assert_allclose(validation, expected_scores, rtol=16 * np.finfo(float).eps, atol=0.0)
    before = (X.tobytes(), model.dual_coef_.tobytes(), model.intercept_.tobytes())
    # Passing real object coordinates exercises the shared real-value guard
    # before the same range recovery and uncertainty reporting are used.
    report = optimization_diagnostics(model, X.astype(object), y, block_size=1)
    assert report["rkhs_norm_squared"] == pytest.approx(expected_norm, rel=16 * np.finfo(float).eps, abs=0.0)
    assert report["rkhs_gradient_norm"] == pytest.approx(expected_gradient, rel=16 * np.finfo(float).eps, abs=0.0)
    risks = report["kernel_matrix_product_uncertainty"]
    assert risks["representation"] == "rbf_expm1_contrast"
    for name in ("coefficient_product", "gradient_product", "dual_product"):
        assert risks[name]["matrix_entry_magnitude"] == pytest.approx(
            expected_magnitude, rel=16 * np.finfo(float).eps, abs=0.0)
    assert report["kernel_passes"] == 2
    assert report["n_kernel_values"] == 8
    assert before == (X.tobytes(), model.dual_coef_.tobytes(), model.intercept_.tobytes())
    json.dumps(risks, allow_nan=False)


def test_literal_psd_inner_cancellation_warns_without_replacing_the_reported_norm():
    n = 17
    X = np.arange(n, dtype=float)[:, None]
    y = np.where(np.arange(n) % 2, 1, -1)
    matrix = np.ones((n, n)) + np.eye(n) * 2.0**-52
    calls = []

    def kernel(left, right):
        calls.append((left[:, 0].tolist(), right[:, 0].tolist()))
        return matrix[np.ix_(left[:, 0].astype(int), right[:, 0].astype(int))]

    model = KernelSGDDWD(kernel=kernel, lambd=2.0**-10, max_epochs=1,
                         stopping="fixed", optimizer_backend="python", random_state=7).fit(X, y)
    model.dual_coef_[:] = 0.0
    model.dual_coef_[0, :2] = [1e8, -1e8]
    model.intercept_[:] = 0.0
    calls.clear()
    original = matrix.tobytes()
    report = optimization_diagnostics(model, X, y, block_size=n)
    exact_norm = Fraction(2) * Fraction.from_float(1e8) ** 2 * Fraction(1, 2**52)
    assert exact_norm > 0
    risks = report["kernel_matrix_product_uncertainty"]
    assert risks["representation"] == "literal_kernel"
    assert risks["coefficient_product"]["quadratic_component_warning"]
    assert risks["coefficient_product"]["matrix_entry_magnitude"] == 1.0 + 2.0**-52
    assert len(calls) == 2
    assert calls[0] == calls[1] == (list(X[:, 0]), list(X[:, 0]))
    assert matrix.tobytes() == original
    assert "not formal error bounds" in risks["qualification"]
    assert "no acceptance or stopping" in risks["qualification"]
    assert np.isfinite(report["rkhs_norm_squared"])
    # The indicator is not a replacement norm: BLAS-dependent values remain
    # visible. A warning cannot be interpreted as a new solver failure.
    assert report["raw_rkhs_norm_squared"] == report["rkhs_norm_squared"]


@pytest.mark.parametrize("case", ["zero_mass", "zero_matrix", "large", "subnormal"])
@pytest.mark.parametrize("policy", ["ignore", "warn", "raise"])
def test_reporting_scales_do_not_mutate_caller_error_policy(case, policy):
    if case == "zero_mass":
        vector, maximum = np.zeros(2), 1.0
    elif case == "zero_matrix":
        vector, maximum = np.array([1e308, -1e308]), 0.0
    elif case == "large":
        vector, maximum = np.array([2.0**600, -2.0**600]), 1.0
    else:
        vector, maximum = np.array([2.0**-1074]), 2.0**-1074
    with np.errstate(all=policy):
        before = np.geterr().copy()
        report = diagnostics._kernel_product_uncertainty(vector, maximum, np.zeros(len(vector)), 0.0, len(vector))
        assert np.geterr() == before
    assert report["quadratic_component_warning"] == (case in ("large", "subnormal"))
    json.dumps(report, allow_nan=False)


def test_diagnostic_custom_distance_failure_is_not_recovered_or_replaced():
    X = np.array([[-5e-163], [5e-163]])
    y = np.array([-1, 1])
    model = KernelSGDDWD(gamma=1e24, max_epochs=1, stopping="fixed",
                         optimizer_backend="python", random_state=7).fit(X, y)
    marker = RuntimeError("custom diagnostic distance failure")
    with patch.object(diagnostics, "cdist", side_effect=marker):
        with pytest.raises(RuntimeError) as caught:
            optimization_diagnostics(model, X, y, block_size=1)
    assert caught.value is marker
