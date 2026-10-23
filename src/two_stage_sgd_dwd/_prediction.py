"""Bounded RBF prediction for already validated public query features.

Public prediction uses the bounded evaluators. Eligible validation checkpoints
also reuse the narrow near-constant score recovery helper; optimizer updates
retain their original kernel evaluation path.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import cdist

from ._core import (_finite_array, _finite_result, _finite_scalar, _scale_rbf_distances,
                    _rbf_zero_distances_are_exact)
from .profiling import _current_call, _profiled

_ORIGINAL_CDIST = cdist
_ORIGINAL_FINITE_RESULT = _finite_result


@_profiled("direct")
def rbf_prediction_scores(X, basis, coefficients, intercept, gamma, batch_size):
    """Evaluate the original direct-distance formula with one reusable block.

    ``X`` has already passed the estimator's public ``validate_data`` call.
    The fitted basis and positive width are checked once per prediction call.
    The caller retains the original routing for zero or nonpositive widths.
    No converted features, distances, or kernels are retained on the estimator.
    """
    profile = _current_call()
    right = _finite_array(basis, "Y", ndim=2)
    if X.shape[1] != right.shape[1] or X.shape[1] == 0:
        raise ValueError("X and Y must have the same positive feature count")
    width = _finite_scalar(gamma, "gamma")
    if width <= 0.0:
        raise ValueError("gamma must be strictly positive")

    # Keep the original loop's interpretation of batch_size. The parameter is
    # validated when fitting; range also rejects non-integral post-fit changes.
    starts = range(0, len(X), batch_size)
    result = np.empty(len(X), dtype=np.float64)
    if not starts:
        return result
    distances = np.empty((min(batch_size, len(X)), len(right)), dtype=np.float64)
    for start in starts:
        if profile is not None:
            profile.start_block()
        block = X[start:start + batch_size]
        # Slicing complete rows preserves C-contiguity, including a short final
        # block. cdist receives exactly the original left/right inputs and shape.
        kernel = distances[:len(block)]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            with np.errstate(under="ignore"):
                cdist(block, right, metric="sqeuclidean", out=kernel)
                if _finite_result is _ORIGINAL_FINITE_RESULT:
                    _scale_rbf_distances(kernel, block, right, width,
                                         trusted=cdist is _ORIGINAL_CDIST)
                else:
                    _finite_result(kernel, "squared Euclidean distances")
                    kernel *= -width
                np.exp(kernel, out=kernel)
            _finite_result(kernel, "RBF kernel")
            if profile is not None:
                profile.phase("score")
            values = kernel @ coefficients + intercept
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite decision scores")
        result[start:start + len(block)] = values
        if profile is not None:
            profile.complete_block(len(block))
    return result


def _direct_distance_rows(block, basis, distances, rows):
    """Replace complete rows in the bounded output; retain its BLAS shape."""
    if np.all(rows):
        cdist(block, basis, metric="sqeuclidean", out=distances)
    else:
        for index in np.flatnonzero(rows):
            cdist(block[index:index + 1], basis, metric="sqeuclidean",
                  out=distances[index:index + 1])


def _distance_rows_to_kernel(distances, width):
    """Apply the same strict exponent arithmetic as direct prediction."""
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        _finite_result(distances, "squared Euclidean distances")
        distances *= -width
        np.exp(distances, out=distances)
    _finite_result(distances, "RBF kernel")

_ORIGINAL_DISTANCE_ROWS_TO_KERNEL = _distance_rows_to_kernel


def _range_checked_distance_rows_to_kernel(distances, width, block, basis):
    """Preserve replaced two-argument hooks; recover built-in distance range."""
    if (_distance_rows_to_kernel is not _ORIGINAL_DISTANCE_ROWS_TO_KERNEL
            or _finite_result is not _ORIGINAL_FINITE_RESULT):
        return _distance_rows_to_kernel(distances, width)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        _scale_rbf_distances(distances, block, basis, width,
                             trusted=cdist is _ORIGINAL_CDIST)
        np.exp(distances, out=distances)
    _finite_result(distances, "RBF kernel")



def _near_constant_scores(block, basis, coefficients, intercept, width,
                          kernel, values, envelope, profile, known_direct):
    """Recover tiny RBF variations only for roundoff-level score cancellation."""
    # exp(-t) - 1 loses about half of binary64's relative precision when
    # t <= sqrt(eps). expm1 retains that variation without changing the formula.
    if profile is not None:
        profile.phase("guard")
    if not np.isfinite(envelope):
        return
    candidates = np.abs(values) <= envelope
    if not np.any(candidates):
        return
    if np.any(known_direct):
        # Only already-direct kernels can reject a row before its probe.
        # Approximate norm/dot rows retain the independent distance check.
        candidates &= (np.logical_not(known_direct) |
                       (np.min(kernel, axis=1) >= 1.0 - 2.0 * np.sqrt(np.finfo(float).eps)))
    candidate_rows = np.flatnonzero(candidates)
    if not len(candidate_rows):
        return
    # Check shared coordinates once per eligible block, not once per probe.
    zero_distances_exact = (cdist is _ORIGINAL_CDIST and
                            _rbf_zero_distances_are_exact(basis) and
                            _rbf_zero_distances_are_exact(block))
    for index in candidate_rows:
        if profile is not None:
            profile.fallback("near_constant_probe", 1)
            profile.phase("kernel")
        row = kernel[index:index + 1]
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            cdist(block[index:index + 1], basis, metric="sqeuclidean", out=row)
            if _finite_result is _ORIGINAL_FINITE_RESULT:
                _scale_rbf_distances(row, block[index:index + 1], basis, width,
                                     trusted=cdist is _ORIGINAL_CDIST, contrast=True,
                                     zero_distances_exact=zero_distances_exact)
            else:
                _finite_result(row, "squared Euclidean distances")
                row *= -width
            if profile is not None:
                profile.complete_fallback("near_constant_probe", 1)
            if np.max(np.abs(row)) > np.sqrt(np.finfo(np.float64).eps):
                continue
            if profile is not None:
                profile.fallback("near_constant_score", 1)
            np.expm1(row, out=row)
            if profile is not None:
                profile.phase("score")
            values[index] = math.fsum([*coefficients, intercept,
                                       *(row[0] * coefficients)])
            _finite_result(values[index:index + 1], "decision scores")
        if profile is not None:
            profile.complete_fallback("near_constant_score", 1)


@_profiled("blas")
def rbf_prediction_scores_blas(X, basis, coefficients, intercept, gamma, batch_size):
    """Norm/dot RBF prediction with conservative direct fallbacks.

    The formula is unchanged, but BLAS reduction order need not reproduce the
    direct helper's score bytes. Dimension-scaled uncertainty envelopes are
    engineering guards, not certified error bounds for every BLAS platform.
    Suspicious distances and uncertain score signs are recomputed directly.
    Near-constant kernels with roundoff-level score cancellation additionally
    use expm1 and accurate summation to preserve tiny RBF variations.
    All buffers are local and at most one prediction block of kernel values is
    retained. ``X`` has already passed the public estimator's validation.
    """
    profile = _current_call()
    right = _finite_array(basis, "Y", ndim=2)
    if X.shape[1] != right.shape[1] or X.shape[1] == 0:
        raise ValueError("X and Y must have the same positive feature count")
    width = _finite_scalar(gamma, "gamma")
    if width <= 0.0:
        raise ValueError("gamma must be strictly positive")
    starts = range(0, len(X), batch_size)
    result = np.empty(len(X), dtype=np.float64)
    if not starts:
        return result

    # Unusual externally modified fitted coefficients retain the direct path's
    # validation/error behavior. Ordinary fitted arrays are finite float64.
    coefficients_array = np.asarray(coefficients)
    if (not len(right) or coefficients_array.dtype != np.dtype(np.float64) or
            coefficients_array.shape != (len(right),) or
            not np.isfinite(coefficients_array).all() or
            np.ndim(intercept) != 0 or not np.isfinite(intercept)):
        return _delegate_direct(profile, "unusual_fitted_state", X, basis, coefficients,
                                intercept, gamma, batch_size)
    info = np.finfo(np.float64)
    distance_roundoff = (X.shape[1] + 3) * info.eps
    score_roundoff = (len(right) + 2) * info.eps
    if max(distance_roundoff, score_roundoff) >= 0.5:
        return _delegate_direct(profile, "roundoff_guard", X, basis, coefficients,
                                intercept, gamma, batch_size)
    distance_factor = 8.0 * distance_roundoff / (1.0 - distance_roundoff)
    score_factor = 8.0 * score_roundoff / (1.0 - score_roundoff)
    # Retain a subnormal allowance rather than using a scale-independent unit
    # floor, which would unnecessarily discard legitimate tiny distances.
    underflow_allowance = info.smallest_subnormal * (8.0 * (X.shape[1] + 3))
    score_underflow_allowance = info.smallest_subnormal * (8.0 * (len(right) + 2))
    absolute_coefficients = np.abs(coefficients_array)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        basis_norms = np.einsum("ij,ij->i", right, right)
        coefficient_mass = float(np.sum(absolute_coefficients))
    if not np.isfinite(basis_norms).all() or not np.isfinite(coefficient_mass):
        return _delegate_direct(profile, "nonfinite_norm_or_mass", X, basis, coefficients,
                                intercept, gamma, batch_size)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        summation_envelope = (4.0 * score_factor * (coefficient_mass + abs(intercept)) +
                              score_underflow_allowance)
    maximum_basis_norm = float(np.max(basis_norms))
    buffer = np.empty((min(batch_size, len(X)), len(right)), dtype=np.float64)

    for start in starts:
        if profile is not None:
            profile.start_block()
        block = X[start:start + batch_size]
        kernel = buffer[:len(block)]
        # Small, low-dimensional blocks avoid norm/dot setup. This execution
        # choice retains strict distance arithmetic and cancellation recovery.
        if (right.size <= 2048 and right.shape[1] <= 32 and len(block) <= 64):
            if profile is not None:
                profile.fallback("execution_direct", len(block))
                profile.phase("kernel")
            cdist(block, right, metric="sqeuclidean", out=kernel)
            _range_checked_distance_rows_to_kernel(kernel, width, block, right)
            if profile is not None:
                profile.phase("score")
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = kernel @ coefficients + intercept
            _finite_result(values, "decision scores")
            _near_constant_scores(block, right, coefficients_array, intercept, width,
                                  kernel, values, summation_envelope, profile, True)
            result[start:start + len(block)] = values
            if profile is not None:
                profile.complete_fallback("execution_direct", len(block))
                profile.complete_block(len(block))
            continue
        # Only approximate intermediates suppress arithmetic exceptions. Any
        # suspicious result is retried by the original strict direct formula.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            norms = np.einsum("ij,ij->i", block, block)
            np.matmul(block, right.T, out=kernel)
            kernel *= -2.0
            kernel += norms[:, None]
            kernel += basis_norms[None, :]
            if profile is not None:
                profile.phase("guard")
            # A row-wide upper envelope avoids a second dense float buffer.
            uncertainty = (distance_factor * norms +
                           distance_factor * maximum_basis_norm + underflow_allowance)
            minimum_distance = np.min(kernel, axis=1)
            direct_rows = (~np.isfinite(kernel).all(axis=1) | ~np.isfinite(uncertainty) |
                           (minimum_distance <= uncertainty))
            if width > 1.0:
                # Preserve strict multiplication overflow near its threshold.
                direct_rows |= np.max(kernel, axis=1) >= info.max / width - uncertainty
        if profile is not None:
            profile.phase("kernel")
        if np.any(direct_rows):
            if profile is not None:
                fallback_rows = int(np.count_nonzero(direct_rows))
                profile.fallback("distance_guard", fallback_rows)
            _direct_distance_rows(block, right, kernel, direct_rows)
            if profile is not None:
                profile.complete_fallback("distance_guard", fallback_rows)
        try:
            _range_checked_distance_rows_to_kernel(kernel, width, block, right)
            if profile is not None:
                profile.phase("score")
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = kernel @ coefficients + intercept
            _finite_result(values, "decision scores")
        except FloatingPointError:
            # A rounded fast intermediate may overflow where direct evaluation
            # is finite. Conversely, genuine direct failures must still raise.
            if profile is not None:
                profile.fallback("arithmetic_retry", len(block))
                profile.phase("kernel")
            _direct_distance_rows(block, right, kernel, np.ones(len(block), dtype=bool))
            _range_checked_distance_rows_to_kernel(kernel, width, block, right)
            if profile is not None:
                profile.phase("score")
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = kernel @ coefficients + intercept
            _finite_result(values, "decision scores")
            _near_constant_scores(block, right, coefficients_array, intercept, width,
                                  kernel, values, summation_envelope, profile, True)
            result[start:start + len(block)] = values
            if profile is not None:
                profile.complete_fallback("arithmetic_retry", len(block))
                profile.complete_block(len(block))
            continue

        if profile is not None:
            profile.phase("guard")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            lower_distance = np.maximum(minimum_distance - uncertainty, 0.0)
            upper_kernel = np.exp(-width * lower_distance)
            kernel_error = (upper_kernel * np.minimum(1.0, width * uncertainty) +
                            16.0 * info.eps)
            absolute_score_terms = kernel @ absolute_coefficients
            score_error = (coefficient_mass * kernel_error +
                           score_factor * (absolute_score_terms + abs(intercept)) +
                           score_underflow_allowance)
            uncertain_scores = (~direct_rows &
                                (~np.isfinite(score_error) | (np.abs(values) <= score_error)))
        known_direct = direct_rows
        if np.any(uncertain_scores):
            # Keep both exp's contiguous length/SIMD tails and the matrix-vector
            # reduction shape identical to direct prediction near the boundary.
            if profile is not None:
                profile.fallback("uncertain_score", len(block), np.count_nonzero(uncertain_scores))
                profile.phase("kernel")
            _direct_distance_rows(block, right, kernel, np.ones(len(block), dtype=bool))
            _range_checked_distance_rows_to_kernel(kernel, width, block, right)
            if profile is not None:
                profile.phase("score")
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = kernel @ coefficients + intercept
            _finite_result(values, "decision scores")
            if profile is not None:
                profile.complete_fallback("uncertain_score", len(block))
            known_direct = True
        _near_constant_scores(block, right, coefficients_array, intercept, width,
                              kernel, values, summation_envelope, profile, known_direct)
        result[start:start + len(block)] = values
        if profile is not None:
            profile.complete_block(len(block))
    return result


def _delegate_direct(profile, reason, X, basis, coefficients, intercept, gamma, batch_size):
    if profile is not None:
        profile.fallback(reason, len(X))
        profile.phase("delegated")
    result = rbf_prediction_scores(X, basis, coefficients, intercept, gamma, batch_size)
    if profile is not None:
        profile.complete_fallback(reason, len(X))
    return result
