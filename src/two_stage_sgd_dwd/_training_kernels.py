"""Fit-local positive-RBF columns for the restricted native optimizer path."""
from __future__ import annotations

import numpy as np

from . import _core


def make_rbf_column_loader(X, gamma):
    """Validate fixed fitting inputs once; compute each requested column afresh.

    The native caller restricts this private helper to ordinary fitting arrays
    and built-in state without user callbacks. Inputs must remain unchanged
    during fitting. No kernel values are computed until the returned callable
    is used, and each call owns a fresh distance array suitable for lazy caching.
    """
    if type(X) is not np.ndarray or X.dtype != np.float64:
        raise TypeError("X must be an ordinary float64 NumPy array")
    left = _core._finite_array(X, "X", ndim=2)
    if left.shape[1] == 0:
        raise ValueError("X and Y must have the same positive feature count")
    width = _core._finite_scalar(gamma, "gamma")
    if width <= 0:
        raise ValueError("gamma must be strictly positive")

    def column(index):
        # Keep rbf_kernel's m-by-1 cdist orientation and arithmetic verbatim.
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            distances = _core.cdist(left, left[index:index + 1], metric="sqeuclidean")
            _core._scale_rbf_distances(distances, left, left[index:index + 1], width,
                                       trusted=_core.cdist is _core._ORIGINAL_CDIST)
            result = np.exp(distances, out=distances)
        _core._finite_result(result, "RBF kernel")
        return result[:, 0]

    return column
