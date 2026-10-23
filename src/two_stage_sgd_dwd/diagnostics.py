"""Optional post-fit optimization diagnostics for the binary DWD estimators.

The dual bound and gap are floating-point numerical estimates, conditional on
the fitted kernel being symmetric PSD. They are not rigorous certificates.
X and y must be the actual fitting observations in their original row order.
Linear estimators do not store those rows, so this remains a caller obligation;
kernel rows are checked exactly against the stored fitting basis.
"""

from __future__ import annotations

import math
import numbers
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils.validation import check_is_fitted

from . import _core, _kernels
from ._core import _contains_complex, _mean_finite_losses
from ._estimators import KernelSGDDWD, LinearSGDDWD, TwoStageDWDClassifier


__all__ = ["optimization_diagnostics"]

_ORIGINAL_KERNEL_EVALUATE = _kernels.KernelSpec.evaluate
_ORIGINAL_RBF_KERNEL = _kernels.rbf_kernel
_ORIGINAL_CDIST = _core.cdist


QUALIFICATION = (
    "Floating-point numerical estimate, not a formally validated optimality "
    "certificate; kernel conclusions assume a deterministic symmetric positive-semidefinite "
    "kernel on the fitting observations. No estimator status is changed."
)


def _array(value, name, ndim):
    raw = np.asarray(value)
    if _contains_complex(raw):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"Nonfinite {name}")
    return result


def _fsum(values):
    try:
        return math.fsum(values)
    except OverflowError as exc:
        raise FloatingPointError("Nonfinite diagnostic sum") from exc


def _norm(value, name):
    # Scaling prevents avoidable overflow/underflow in the squared norm.
    scale = float(np.max(np.abs(value), initial=0.0))
    if scale == 0.0:
        return 0.0
    return _finite(scale * np.linalg.norm(value / scale), name)


def _quadratic(vector, product, name, *, constant_component=None):
    terms = vector * product
    if constant_component is not None:
        # For K = 11' + E, keep (sum v)**2 separate from v' E v.
        # Recombining K v first can erase its small, important contrasts.
        terms = np.r_[np.float64(constant_component) ** 2, terms]
    value = _finite(_fsum(map(float, terms)), name)
    scale = _finite(_fsum(map(float, np.abs(terms))), name + " scale")
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, scale)
    if value < -tolerance:
        raise FloatingPointError(f"{name} is negative beyond roundoff; PSD assumption failed")
    return max(0.0, value), value, tolerance


def _uncertainty_product(*factors):
    """Represent a nonnegative reporting scale without avoidable overflow.

    This supports an indicator, not an outward-rounded error certificate.
    A positive scale below the representable range is marked explicitly.
    """
    if any(value == 0.0 for value in factors):
        return 0.0, "zero"
    mantissa, exponent = 1.0, 0
    for value in factors:
        if not math.isfinite(value) or value < 0.0:
            return None, "unavailable"
        part, power = math.frexp(value)
        mantissa *= part
        mantissa, adjust = math.frexp(mantissa)
        exponent += power + adjust
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError:
        return None, "overflow"
    if result == 0.0:
        return math.ulp(0.0), "underflow_limited"
    return result, "estimated"


def _uncertainty_sum(*scales):
    if any(value is None for value, status in scales):
        return None, "unavailable"
    try:
        value = math.fsum(item[0] for item in scales)
    except OverflowError:
        return None, "overflow"
    status = "underflow_limited" if any(item[1] == "underflow_limited" for item in scales) else "estimated"
    return value, "zero" if value == 0.0 else status


def _kernel_product_uncertainty(vector, maximum, product, quadratic, n):
    """Qualify inner-product risk using M*||v||_1 and M*||v||_1**2.

    M is the observed maximum absolute entry of the actual matrix used by
    this pass. These upper surrogates cover |K||v| and |v|'|K||v| in exact
    arithmetic. Their floating evaluation is a qualified risk indicator.
    """
    magnitude = np.abs(vector)
    largest = float(np.max(magnitude, initial=0.0))
    if largest == 0.0 or maximum == 0.0:
        inner = propagated = (0.0, "zero")
    else:
        with np.errstate(under="ignore"):
            mass = math.fsum(map(float, magnitude / largest))
        neps = n * np.finfo(float).eps
        if neps >= 0.5:
            inner = propagated = (None, "unavailable")
        else:
            factor = neps / (1.0 - neps)
            # eps is twice unit roundoff. Include a gradual-underflow scale
            # without making a portability or directed-rounding guarantee.
            tiny = (2.0 * n + 2.0) * math.ulp(0.0)
            inner = _uncertainty_sum(_uncertainty_product(factor, maximum, largest, mass), (tiny, "estimated"))
            propagated = _uncertainty_sum(
                _uncertainty_product(factor, maximum, largest, mass, largest, mass),
                _uncertainty_product(tiny, largest, mass))
    def warning(scale, reference):
        value, status = scale
        return bool(value is None or status == "underflow_limited"
                    or value > 1e-8 * max(1.0, abs(reference)))
    observed = float(np.max(np.abs(product), initial=0.0))
    return {
        "matrix_entry_magnitude": maximum,
        "maximum_product_error_scale": inner[0],
        "maximum_product_scale_status": inner[1],
        "quadratic_propagation_error_scale": propagated[0],
        "quadratic_scale_status": propagated[1],
        "matrix_product_warning": warning(inner, observed),
        "quadratic_component_warning": warning(propagated, quadratic),
    }


def _inputs(model, X, y, *, require_intercept=True):
    check_is_fitted(model)
    X = _array(X, "X", 2)
    if min(X.shape) <= 0:
        raise ValueError("X must contain fitting rows and features")
    if int(getattr(model, "n_samples_fit_", -1)) != len(X):
        raise ValueError("X row count must equal the actual fitting sample count")
    if int(getattr(model, "n_features_in_", -1)) != X.shape[1]:
        raise ValueError("X feature count does not match the fitted model")
    classes = np.asarray(getattr(model, "classes_", []))
    labels = np.asarray(y)
    if classes.ndim != 1 or len(classes) != 2 or classes[0] == classes[1]:
        raise ValueError("Exactly two distinct fitted classes are required")
    if np.issubdtype(classes.dtype, np.number) and not np.isfinite(classes).all():
        raise ValueError("Fitted numerical classes must be finite")
    if labels.ndim != 1 or len(labels) != len(X):
        raise ValueError("y must have one label per fitting observation")
    negative, positive = labels == classes[0], labels == classes[1]
    if not np.all(negative | positive) or not negative.any() or not positive.any():
        raise ValueError("y must contain both fitted classes and no other labels")
    signed = np.where(positive, 1.0, -1.0)
    if not require_intercept:
        return X, signed, 0.0
    intercept = _array(getattr(model, "intercept_", []), "intercept_", 1)
    if intercept.shape != (1,):
        raise ValueError("Exactly one free intercept is required")
    return X, signed, float(intercept[0])


def _loss_and_alpha(margins):
    if not np.isfinite(margins).all():
        raise FloatingPointError("Nonfinite margins")
    upper = margins > 0.5
    loss, alpha = np.empty_like(margins), np.ones_like(margins)
    loss[~upper] = 1.0 - margins[~upper]
    loss[upper] = 0.25 / margins[upper]
    alpha[upper] = (0.5 / margins[upper]) ** 2
    try:
        mean_loss = math.fsum(map(float, loss)) / len(loss)
    except OverflowError:
        # Only the sum overflowed: a mean of finite nonnegative losses can
        # still be represented. Preserve the compensated ordinary reduction.
        mean_loss = _mean_finite_losses(loss)
    return _finite(mean_loss, "mean loss"), alpha


def _near_constant_rbf_width(spec, X):
    """Qualify only the unchanged built-in RBF on a narrow feature range.

    This execution guard is not an error bound. Custom kernel methods, even
    ones advertising an RBF name, retain their original evaluation behavior.
    """
    if (type(spec) is not _kernels.KernelSpec
            or getattr(spec.evaluate, "__func__", None) is not _ORIGINAL_KERNEL_EVALUATE
            or _kernels.rbf_kernel is not _ORIGINAL_RBF_KERNEL
            or _core.cdist is not _ORIGINAL_CDIST
            or type(spec.kernel) is not str
            or spec.kernel != "rbf"):
        return None
    width = spec.gamma
    if (isinstance(width, (bool, np.bool_)) or not isinstance(width, numbers.Real)
            or not math.isfinite(width) or width <= 0.0):
        return None
    # resolve_kernel stores gamma in kws; hand-constructed empty specs are also
    # valid. Do not bypass inconsistent or unexpected fitted keyword state.
    if type(spec.kws) is not dict or set(spec.kws) - {"gamma"}:
        return None
    keyword_width = spec.kws.get("gamma", width)
    if (isinstance(keyword_width, (bool, np.bool_))
            or not isinstance(keyword_width, numbers.Real)
            or keyword_width != width):
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            spans = np.max(X, axis=0) - np.min(X, axis=0)
            bound = float(np.sum(spans * spans)) * width
    except FloatingPointError:
        return None
    return float(width) if math.isfinite(bound) and bound <= math.sqrt(np.finfo(float).eps) else None


def _balanced_alpha(raw_alpha, signed):
    alpha = raw_alpha.copy()
    groups = signed > 0.0, signed < 0.0
    totals = [_fsum(map(float, alpha[mask])) for mask in groups]
    high = 0 if totals[0] >= totals[1] else 1
    if totals[high] > 0.0:
        alpha[groups[high]] *= totals[1 - high] / totals[high]
    # One extra subtraction reduces residual imbalance without increasing any
    # coordinate. Residual float error is explicitly bounded below, not hidden.
    q = signed * alpha / len(alpha)
    residual = _fsum(map(float, q))
    if residual != 0.0:
        mask = signed == (1.0 if residual > 0.0 else -1.0)
        candidates = np.flatnonzero(mask)
        pivot = candidates[np.argmax(alpha[candidates])]
        alpha[pivot] = max(0.0, alpha[pivot] - abs(residual) * len(alpha))
    q = signed * alpha / len(alpha)
    return alpha, q, _fsum(map(float, q))


def _result(kind, objective, raw_alpha, alpha, q, balance, intercept_bound,
            dual_unadjusted, gb, stationarity, started, **details):
    correction = _finite(intercept_bound * abs(balance), "balance correction")
    corrected = _finite(dual_unadjusted - correction, "corrected dual bound")
    # Zero is an independently feasible dual point with objective zero.
    bound = max(0.0, corrected)
    gap = _finite(objective - bound, "numerical gap estimate")
    gap_tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(objective), abs(bound))
    if gap < -gap_tolerance:
        raise FloatingPointError("Negative numerical gap beyond roundoff; diagnostic is invalid")
    return {
        "kind": kind,
        "objective": float(objective),
        "numerical_dual_bound_estimate": bound,
        "dual_objective_before_balance_correction": float(dual_unadjusted),
        "dual_balance_correction": correction,
        "dual_balance_residual": float(balance),
        "dual_box_violation": float(max(0.0, -np.min(alpha), np.max(alpha) - 1.0)),
        "numerical_gap_estimate": gap,
        "negative_gap_observed": bool(gap < 0.0),
        "negative_gap_roundoff_tolerance": gap_tolerance,
        "dual_bound_source": "balanced_derivative_candidate" if corrected >= 0.0 else "zero_dual_candidate",
        "optimal_intercept_absolute_bound": float(intercept_bound),
        "free_intercept_derivative": float(gb),
        "stationarity_residual": float(stationarity),
        "raw_loss_dual_alpha_range": [float(np.min(raw_alpha)), float(np.max(raw_alpha))],
        "qualification": QUALIFICATION,
        "formal_certificate": False,
        "elapsed_seconds": time.perf_counter() - started,
        **details,
    }


def _linear(model, X, signed, intercept, started):
    coef = _array(getattr(model, "coef_", []), "coef_", 2)
    if coef.shape != (1, X.shape[1]):
        raise ValueError("Linear coefficient dimensions do not match X")
    w = coef[0]
    wn = _norm(w, "slope norm")
    violation = max(0.0, wn - 1.0)
    if violation > 64.0 * np.finfo(float).eps:
        raise ValueError("Linear slope violates the fitted unit-ball constraint")
    objective, raw_alpha = _loss_and_alpha(signed * (X @ w + intercept))
    raw_q = signed * raw_alpha / len(X)
    gw, gb = -(X.T @ raw_q), -_fsum(map(float, raw_q))
    step = w - gw
    projection = step / max(1.0, _norm(step, "gradient step norm"))
    projected = _norm(w - projection, "projected gradient residual")
    alpha, q, balance = _balanced_alpha(raw_alpha, signed)
    dual_vector = X.T @ q
    dual_penalty = _norm(dual_vector, "dual norm")
    dual = _fsum(map(float, np.sqrt(alpha))) / len(X) - dual_penalty
    max_row_norm = max(_norm(row, "row norm") for row in X)
    minority = min(np.count_nonzero(signed > 0), np.count_nonzero(signed < 0))
    intercept_bound = _finite(len(X) / minority - 1.0 + max_row_norm, "intercept bound")
    return _result(
        "linear", objective, raw_alpha, alpha, q, balance, intercept_bound,
        dual, gb, math.hypot(projected, gb), started,
        n_samples=len(X), n_features=X.shape[1], unit_ball_violation=violation,
        projected_gradient_norm=projected, projected_gradient_step_size=1.0,
        n_kernel_values=0, max_block_elements=0,
        fitting_rows_verified_against_model=False,
        fitting_rows_requirement="Caller must supply the original linear fitting rows and labels.",
    )


def _kernel(model, X, signed, intercept, block_size, started):
    basis = _array(getattr(model, "basis_", []), "basis_", 2)
    if basis.shape != X.shape or not np.array_equal(basis, X):
        raise ValueError("Kernel X must exactly match actual fitting basis in row order")
    coef = _array(getattr(model, "dual_coef_", []), "dual_coef_", 2)
    if coef.shape != (1, len(X)):
        raise ValueError("Kernel coefficients must match the fitting basis")
    lambd = getattr(model, "lambd_", None)
    if isinstance(lambd, (bool, np.bool_)) or not isinstance(lambd, numbers.Real) or not np.isfinite(lambd) or lambd <= 0.0:
        raise ValueError("Fitted lambd_ must be finite and positive")
    spec = getattr(model, "_kernel_spec_", None)
    if not callable(getattr(spec, "evaluate", None)):
        raise ValueError("A fitted kernel specification is required")
    a, n = coef[0], len(X)
    contrast_width = _near_constant_rbf_width(spec, X)
    # Every diagnostic block is a view of this same validated fitting matrix.
    zero_distances_exact = (contrast_width is not None and cdist is _ORIGINAL_CDIST and
                            _core._rbf_zero_distances_are_exact(X))
    Ka = np.empty(n)
    max_diagonal, evaluated, maximum_block = 0.0, 0, 0
    maximum_entries = [0.0, 0.0]
    pass_index = 0

    def block(start):
        nonlocal evaluated, maximum_block
        stop = min(n, start + block_size)
        if contrast_width is None:
            values = _array(spec.evaluate(X[start:stop], X), "kernel block", 2)
        else:
            # Evaluate E = K - 11' directly, without rounding K to one first.
            # These are the same n-by-n kernel entries represented as contrasts;
            # each is evaluated once per pass, with no additional Gram pass.
            values = cdist(X[start:stop], X, metric="sqeuclidean")
            _core._scale_rbf_distances(values, X[start:stop], X, contrast_width,
                                       trusted=cdist is _ORIGINAL_CDIST, contrast=True,
                                       zero_distances_exact=zero_distances_exact)
            np.expm1(values, out=values)
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite RBF kernel contrasts")
        if values.shape != (stop - start, n):
            raise ValueError("Kernel block shape does not match fitting observations")
        maximum_entries[pass_index] = max(
            maximum_entries[pass_index], abs(float(np.min(values, initial=0.0))),
            abs(float(np.max(values, initial=0.0))))
        evaluated += values.size
        maximum_block = max(maximum_block, values.size)
        return stop, values

    for start in range(0, n, block_size):
        stop, values = block(start)
        Ka[start:stop] = values @ a
        diagonal = values[np.arange(stop - start), np.arange(start, stop)]
        if contrast_width is not None:
            diagonal = diagonal + 1.0
        if np.any(diagonal < 0.0):
            raise FloatingPointError("Negative kernel diagonal violates PSD assumption")
        max_diagonal = max(max_diagonal, float(np.max(diagonal)))
        del values
    coefficient_sum = None if contrast_width is None else _fsum(map(float, a))
    norm_squared, raw_norm_squared, norm_tolerance = _quadratic(
        a, Ka, "RKHS squared norm", constant_component=coefficient_sum)
    norm_uncertainty = _kernel_product_uncertainty(
        a, maximum_entries[0], Ka, raw_norm_squared, n)
    if contrast_width is not None:
        Ka += coefficient_sum
    mean_loss, raw_alpha = _loss_and_alpha(signed * (Ka + intercept))
    objective = _finite(mean_loss + lambd * norm_squared, "kernel objective")
    raw_q = signed * raw_alpha / n
    t = 2.0 * lambd * a - raw_q
    gb = -_fsum(map(float, raw_q))
    alpha, q, balance = _balanced_alpha(raw_alpha, signed)
    # Both products share the second block traversal; no persistent Gram matrix.
    products = np.empty((n, 2))
    vectors = np.column_stack((t, q))
    pass_index = 1
    for start in range(0, n, block_size):
        stop, values = block(start)
        products[start:stop] = values @ vectors
        del values
    gradient_sum = None if contrast_width is None else _fsum(map(float, t))
    dual_sum = None if contrast_width is None else _fsum(map(float, q))
    gradient_squared, raw_gradient_squared, gradient_tolerance = _quadratic(
        t, products[:, 0], "RKHS gradient squared norm", constant_component=gradient_sum)
    dual_squared, raw_dual_squared, dual_tolerance = _quadratic(
        q, products[:, 1], "dual RKHS squared norm", constant_component=dual_sum)
    dual_penalty = _finite((dual_squared / 4.0) / lambd, "dual penalty")
    dual = _finite(_fsum(map(float, np.sqrt(alpha))) / n - dual_penalty, "dual objective")
    dual_roundoff_scale = _finite((dual_tolerance / 4.0) / lambd, "dual penalty roundoff scale")
    minority = min(np.count_nonzero(signed > 0), np.count_nonzero(signed < 0))
    intercept_bound = _finite(n / minority - 1.0 + math.sqrt(max_diagonal) / math.sqrt(lambd), "intercept bound")
    gradient = math.sqrt(gradient_squared)
    return _result(
        "kernel", objective, raw_alpha, alpha, q, balance, intercept_bound,
        dual, gb, math.hypot(gradient, gb), started,
        n_samples=n, n_features=X.shape[1], lambd=float(lambd),
        rkhs_norm_squared=norm_squared, rkhs_gradient_norm=gradient,
        raw_rkhs_norm_squared=raw_norm_squared,
        raw_rkhs_gradient_squared=raw_gradient_squared,
        raw_dual_rkhs_squared=raw_dual_squared,
        quadratic_roundoff_tolerances=[norm_tolerance, gradient_tolerance, dual_tolerance],
        kernel_matrix_product_uncertainty={
            "representation": "literal_kernel" if contrast_width is None else "rbf_expm1_contrast",
            "coefficient_product": norm_uncertainty,
            "gradient_product": _kernel_product_uncertainty(
                t, maximum_entries[1], products[:, 0], raw_gradient_squared, n),
            "dual_product": _kernel_product_uncertainty(
                q, maximum_entries[1], products[:, 1], raw_dual_squared, n),
            "reporting_relative_threshold": 1e-8,
            "reporting_absolute_reference_floor": 1.0,
            "qualification": (
                "Qualified matrix-product risk indicators, not formal error bounds or "
                "certificates. M*||v||_1 and M*||v||_1**2 are conservative exact-arithmetic "
                "surrogates for |K||v| and |v|'|K||v|; their evaluated scales can be "
                "pessimistic. They exclude kernel-construction error, separately summed "
                "RBF constant components and propagated errors in forming the gradient "
                "or dual vectors. Product errors can also affect loss and alpha. "
                "Thresholds are reporting choices only; no acceptance or stopping "
                "decision changes. A false warning does not establish accuracy. "
                "Unavailable or underflow-limited scales warn without altering the result."
            ),
        },
        dual_penalty_roundoff_scale=dual_roundoff_scale,
        dual_bound_roundoff_warning=bool(dual_roundoff_scale > 1e-8 * max(1.0, abs(objective))),
        roundoff_qualification=(
            "Quadratic roundoff scales are numerical checks, not bounds on all "
            "kernel or matrix-product error. dual_bound_roundoff_warning marks "
            "potential amplification by weak regularization, including zero or "
            "positive quadratics; it need not imply an actual error. A false "
            "warning does not rule out cancellation or certify the dual bound."
        ),
        n_kernel_values=int(evaluated), max_block_elements=int(maximum_block),
        kernel_passes=2, block_size=int(block_size),
        fitting_rows_verified_against_model=True,
        fitting_labels_requirement="Caller must supply the original fitting labels in basis order.",
    )


def optimization_diagnostics(estimator, X, y, *, block_size=256):
    """Evaluate optional numerical optimization diagnostics after fitting.

    Parameters
    ----------
    estimator : LinearSGDDWD, KernelSGDDWD, or TwoStageDWDClassifier
        A fitted binary estimator. For multiclass wrappers, call this helper on
        each fitted binary estimator with its actual fitting observations.
    X : array-like of shape (n_samples, n_features)
        The original fitting feature matrix, including any preprocessing used
        during fitting, in its original row order. A two-stage estimator requires
        the full original matrix; the helper selects its retained rows itself.
    y : array-like of shape (n_samples,)
        The original fitting labels, in the same order and label encoding.
    block_size : int, default=256
        Maximum rows in each evaluated kernel block. Kernel diagnostics make two
        passes over the fitting kernel, evaluating ``2 * n_samples**2`` values,
        with O(block_size * n_samples) kernel working storage. The two-stage
        kernel sample count is the number of retained observations. No Gram
        matrix is retained after this call. Qualified near-constant RBF kernels
        evaluate those entries as ``expm1(-gamma * distance_squared)`` contrasts
        and account separately for the constant component in each quadratic.

    Returns
    -------
    report : dict
        JSON-compatible objective, numerical dual-bound and gap estimates,
        stationarity residuals, elapsed time, and kernel work counts. The linear
        residual uses a unit-step projected gradient on the slope's unit ball;
        the kernel residual uses the RKHS gradient norm. Each includes the free
        intercept derivative. Two-stage reports contain ``linear_stage`` and
        ``kernel_stage`` dictionaries, each normalized by its own fitting size.

    Raises
    ------
    TypeError
        If the estimator is not one of the supported binary estimator types.
    sklearn.exceptions.NotFittedError
        If the estimator or a required child estimator is not fitted.
    ValueError
        If the supplied data, fitted metadata, or block size are inconsistent.
    FloatingPointError
        If a nonfinite calculation or materially invalid numerical bound is
        encountered. Tiny negative gaps remain visible as roundoff diagnostics.

    Notes
    -----
    This helper runs only when explicitly called. It does not update parameters,
    RNG state, stopping criteria, or fitted convergence and optimality flags.
    Both stage intercepts remain unrestricted and unpenalized. Its dual bound is
    a floating-point candidate estimate, which may be conservative; it is not a
    formally validated certificate or a new automatic stopping rule. Rounded
    dual coefficients and their balance correction do not establish exact dual
    feasibility.

    Kernel rows are checked exactly against the stored fitting basis. Original
    linear rows and original fitting labels are not retained by the estimators:
    same-sized replacement linear data or relabeled observations cannot always
    be detected. Supplying the actual fitting data is the caller's responsibility.

    Kernel conclusions assume a deterministic symmetric PSD kernel on the
    fitting observations. The helper does not repeat a full PSD validation.
    User-provided callable kernels can have external side effects outside the
    helper's control. Roundoff indicators are numerical checks, not rigorous
    error bounds; cancellation and very weak regularization can make a bound
    unreliable even when no warning flag is set.
    """
    started = time.perf_counter()
    model = estimator
    if not isinstance(model, (LinearSGDDWD, KernelSGDDWD, TwoStageDWDClassifier)):
        raise TypeError("estimator must be LinearSGDDWD, KernelSGDDWD, or TwoStageDWDClassifier")
    if isinstance(block_size, (bool, np.bool_)) or not isinstance(block_size, numbers.Integral) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if isinstance(model, TwoStageDWDClassifier):
        X, _, _ = _inputs(model, X, y, require_intercept=False)
        selected = np.asarray(getattr(model, "selected_indices_", []))
        if (selected.ndim != 1 or selected.dtype.kind not in "iu" or len(selected) == 0
                or np.any(selected < 0) or np.any(selected >= len(X))
                or len(np.unique(selected)) != len(selected)):
            raise ValueError("Two-stage selected_indices_ must be unique valid fitting row indices")
        if int(getattr(model, "n_retained_", -1)) != len(selected):
            raise ValueError("Two-stage selected size disagrees with fitted metadata")
        linear, kernel = getattr(model, "linear_estimator_", None), getattr(model, "kernel_estimator_", None)
        if not isinstance(linear, LinearSGDDWD) or not isinstance(kernel, KernelSGDDWD):
            raise ValueError("Two-stage child estimators do not match the fitted stage types")
        if not (np.array_equal(getattr(linear, "classes_", []), model.classes_)
                and np.array_equal(getattr(kernel, "classes_", []), model.classes_)):
            raise ValueError("Two-stage class labels disagree")
        first = optimization_diagnostics(linear, X, y, block_size=block_size)
        second = optimization_diagnostics(kernel, X[selected], np.asarray(y)[selected], block_size=block_size)
        return {
            "kind": "two_stage", "linear_stage": first, "kernel_stage": second,
            "normalization": "Each stage averages over its own actual fitting set.",
            "elapsed_seconds": time.perf_counter() - started,
            "n_kernel_values": second["n_kernel_values"],
            "max_block_elements": second["max_block_elements"],
            "formal_certificate": False, "qualification": QUALIFICATION,
        }
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        X, signed, intercept = _inputs(model, X, y)
        if isinstance(model, KernelSGDDWD):
            return _kernel(model, X, signed, intercept, int(block_size), started)
        if isinstance(model, LinearSGDDWD):
            return _linear(model, X, signed, intercept, started)
        raise ValueError("Unsupported fitted estimator: expected linear, kernel, or two-stage SGD DWD")
