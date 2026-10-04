"""Bounded RBF prediction for already validated public query features.

Only public prediction opts into this helper. Training, validation checkpoints,
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
