"""Bounded RBF prediction for already validated public query features.

Only public prediction uses these helpers. Training, validation checkpoints,
deadlines, and kernel diagnostics retain their original evaluation path.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ._core import _finite_array, _finite_result, _finite_scalar


def rbf_prediction_scores(X, basis, coefficients, intercept, gamma, batch_size):
    """Evaluate the original direct-distance formula with one reusable block.

    ``X`` has already passed the estimator's public ``validate_data`` call.
    The fitted basis and positive width are checked once per prediction call.
    The caller retains the original routing for zero or nonpositive widths.
    No converted features, distances, or kernels are retained on the estimator.
    """
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
        block = X[start:start + batch_size]
        # Slicing complete rows preserves C-contiguity, including a short final
        # block. cdist receives exactly the original left/right inputs and shape.
        kernel = distances[:len(block)]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            with np.errstate(under="ignore"):
                cdist(block, right, metric="sqeuclidean", out=kernel)
                _finite_result(kernel, "squared Euclidean distances")
                kernel *= -width
                np.exp(kernel, out=kernel)
            _finite_result(kernel, "RBF kernel")
            values = kernel @ coefficients + intercept
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite decision scores")
        result[start:start + len(block)] = values
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


def rbf_prediction_scores_blas(X, basis, coefficients, intercept, gamma, batch_size):
    """Norm/dot RBF prediction with conservative direct fallbacks.

    The formula is unchanged, but BLAS reduction order need not reproduce the
    direct helper's score bytes. Dimension-scaled uncertainty envelopes are
    engineering guards, not certified error bounds for every BLAS platform.
    Suspicious distances and uncertain score signs are recomputed directly.
    All buffers are local and at most one prediction block of kernel values is
    retained. ``X`` has already passed the public estimator's validation.
    """
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
        return rbf_prediction_scores(X, basis, coefficients, intercept, gamma, batch_size)
    info = np.finfo(np.float64)
    distance_roundoff = (X.shape[1] + 3) * info.eps
    score_roundoff = (len(right) + 2) * info.eps
    if max(distance_roundoff, score_roundoff) >= 0.5:
        return rbf_prediction_scores(X, basis, coefficients, intercept, gamma, batch_size)
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
        return rbf_prediction_scores(X, basis, coefficients, intercept, gamma, batch_size)
    maximum_basis_norm = float(np.max(basis_norms))
    buffer = np.empty((min(batch_size, len(X)), len(right)), dtype=np.float64)

    for start in starts:
        block = X[start:start + batch_size]
        kernel = buffer[:len(block)]
        # Only approximate intermediates suppress arithmetic exceptions. Any
        # suspicious result is retried by the original strict direct formula.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            norms = np.einsum("ij,ij->i", block, block)
            np.matmul(block, right.T, out=kernel)
            kernel *= -2.0
            kernel += norms[:, None]
            kernel += basis_norms[None, :]
            # A row-wide upper envelope avoids a second dense float buffer.
            uncertainty = (distance_factor * norms +
                           distance_factor * maximum_basis_norm + underflow_allowance)
            minimum_distance = np.min(kernel, axis=1)
            direct_rows = (~np.isfinite(kernel).all(axis=1) | ~np.isfinite(uncertainty) |
                           (minimum_distance <= uncertainty))
            if width > 1.0:
                # Preserve strict multiplication overflow near its threshold.
                direct_rows |= np.max(kernel, axis=1) >= info.max / width - uncertainty
        if np.any(direct_rows):
            _direct_distance_rows(block, right, kernel, direct_rows)
        try:
            _distance_rows_to_kernel(kernel, width)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = kernel @ coefficients + intercept
            _finite_result(values, "decision scores")
        except FloatingPointError:
            # A rounded fast intermediate may overflow where direct evaluation
            # is finite. Conversely, genuine direct failures must still raise.
            _direct_distance_rows(block, right, kernel, np.ones(len(block), dtype=bool))
            _distance_rows_to_kernel(kernel, width)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = kernel @ coefficients + intercept
            _finite_result(values, "decision scores")
            result[start:start + len(block)] = values
            continue

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
        if np.any(uncertain_scores):
            # Keep both exp's contiguous length/SIMD tails and the matrix-vector
            # reduction shape identical to direct prediction near the boundary.
            _direct_distance_rows(block, right, kernel, np.ones(len(block), dtype=bool))
            _distance_rows_to_kernel(kernel, width)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = kernel @ coefficients + intercept
            _finite_result(values, "decision scores")
        result[start:start + len(block)] = values
    return result
