"""Small numerical operations for the Chapter 3 two-stage DWD method.

The step functions never mutate their inputs. Kernel coefficients are signed:
the training label is included in each coefficient increment, not in prediction.
"""

from __future__ import annotations

import math
import operator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist

_ORIGINAL_CDIST = cdist


def _contains_complex(array):
    """Inspect represented object entries without invoking conversion hooks."""
    if array.dtype.kind != "O":
        return array.dtype.kind == "c"
    pending, seen = [array], set()
    complex_types, array_type = (complex, np.complexfloating), np.ndarray
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for item in current.flat:
            item_type = type(item)
            # Exact built-in real types cannot hide an imaginary component;
            # subclasses and custom conversion objects retain full inspection.
            if item_type is float or item_type is int or item_type is bool:
                continue
            if issubclass(item_type, complex_types):
                return True
            if issubclass(item_type, array_type):
                if item.dtype.kind == "c":
                    return True
                if item.dtype.kind == "O":
                    pending.append(item)
    return False


def _finite_array(value: ArrayLike, name: str, ndim: int | None = None) -> NDArray[np.float64]:
    try:
        supplied = np.asarray(value)
        if _contains_complex(supplied):
            raise ValueError(f"{name} must contain real numerical values")
        result = np.asarray(supplied, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numerical values") from exc
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_scalar(value: float, name: str) -> float:
    result = _finite_array(value, name, ndim=0)
    return float(result)


def _finite_result(value: ArrayLike, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} produced a nonfinite value")


def _label(value: float) -> float:
    result = _finite_scalar(value, "y")
    if result not in (-1.0, 1.0):
        raise ValueError("y must be -1 or +1")
    return result


def _step_size(value: float) -> float:
    result = _finite_scalar(value, "eta")
    if result < 0.0:
        raise ValueError("eta must be nonnegative")
    return result


def dwd_loss(u: ArrayLike) -> float | NDArray[np.float64]:
    """Normalized DWD loss, with its reciprocal evaluated only for ``u > .5``.

    Scalar input produces a Python float. Array input produces a new float64
    array of the same shape. Nonfinite inputs are rejected.
    """
    margins = _finite_array(u, "u")
    result = np.empty_like(margins)
    upper = margins > 0.5
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        result[~upper] = 1.0 - margins[~upper]
        result[upper] = 0.25 / margins[upper]
    _finite_result(result, "DWD loss")
    return float(result) if margins.ndim == 0 else result


def dwd_loss_derivative(u: ArrayLike) -> float | NDArray[np.float64]:
    """Derivative of :func:`dwd_loss`, including its value -1 at ``u=.5``."""
    margins = _finite_array(u, "u")
    result = np.full_like(margins, -1.0)
    upper = margins > 0.5
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        upper_margins = margins[upper]
        result[upper] = (-0.25 / upper_margins) / upper_margins
    _finite_result(result, "DWD derivative")
    return float(result) if margins.ndim == 0 else result


def _mean_finite_losses(losses: ArrayLike) -> float:
    """Average finite nonnegative losses without overflowing a finite mean.

    Preserve ordinary reduction order whenever its sum is representable.
    Reduction overflow is an internal event, not an output overflow, so it
    does not invoke the caller's warning, callback or exception policy.
    """
    values = np.asarray(losses)
    if (values.ndim != 1 or not values.size or values.dtype.kind not in "biuf"
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("losses must be a nonempty vector of finite nonnegative numbers")
    with np.errstate(over="ignore"):
        value = float(np.mean(values))
    if not np.isfinite(value):
        scale = float(np.max(values))
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            value = scale * float(np.mean(values / scale))
    _finite_result(value, "mean DWD loss")
    return value


def project_unit_ball(w: ArrayLike) -> NDArray[np.float64]:
    """Return the Euclidean projection of a vector onto the closed unit ball.

    Rescaling before computing a large vector's norm avoids overflow. Even
    inside the ball, the return value is an independent array.
    """
    weights = _finite_array(w, "w", ndim=1)
    scale = float(np.max(np.abs(weights), initial=0.0))
    if scale > 1.0:
        scaled = weights / scale
        result = scaled / np.linalg.norm(scaled)
    else:
        norm = float(np.linalg.norm(weights))
        result = weights / norm if norm > 1.0 else weights.copy()
    _finite_result(result, "unit-ball projection")
    return result


def linear_step(
    w: ArrayLike, b: float, x: ArrayLike, y: float, eta: float
) -> tuple[NDArray[np.float64], float]:
    """One projected linear DWD step using the margin from the old state.

    Only ``w`` is projected; ``b`` remains scalar and unregularized.
    """
    weights = _finite_array(w, "w", ndim=1)
    features = _finite_array(x, "x", ndim=1)
    if weights.shape != features.shape:
        raise ValueError("w and x must have the same shape")
    intercept = _finite_scalar(b, "b")
    label = _label(y)
    step = _step_size(eta)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        margin = label * (float(np.dot(weights, features)) + intercept)
        _finite_result(margin, "linear margin")
        q = float(dwd_loss_derivative(margin)) * label
        proposal = weights - (step * q) * features
        new_intercept = intercept - step * q
    _finite_result(proposal, "linear weight update")
    _finite_result(new_intercept, "linear intercept update")
    return project_unit_ball(proposal), float(new_intercept)


def kernel_step(
    a: ArrayLike,
    b: float,
    k: ArrayLike,
    i: int,
    y_i: float,
    eta: float,
    lambd: float,
) -> tuple[NDArray[np.float64], float]:
    """One RKHS DWD step represented by signed expansion coefficients.

    The objective is mean DWD loss plus ``lambd * ||h||^2``. ``k[j]`` is
    ``K(x[j], x[i])``. All coefficients shrink by ``1-2*eta*lambd`` and only
    coefficient ``i`` receives the sampled
    loss increment. The scalar intercept has no penalty. The caller determines
    the step schedule; the Chapter 3 finite-iterate norm bound additionally
    requires ``2*eta*lambd <= 1``.
    """
    coefficients = _finite_array(a, "a", ndim=1)
    kernel_column = _finite_array(k, "k", ndim=1)
    if coefficients.shape != kernel_column.shape:
        raise ValueError("a and k must have the same shape")
    try:
        index = operator.index(i)
    except TypeError as exc:
        raise ValueError("i must be an integer index") from exc
    if isinstance(i, (bool, np.bool_)) or not 0 <= index < coefficients.size:
        raise ValueError("i must index an existing kernel coefficient")
    intercept = _finite_scalar(b, "b")
    label = _label(y_i)
    step = _step_size(eta)
    regularization = _finite_scalar(lambd, "lambd")
    if regularization <= 0.0:
        raise ValueError("lambd must be strictly positive")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        margin = label * (float(np.dot(coefficients, kernel_column)) + intercept)
        _finite_result(margin, "kernel margin")
        q = float(dwd_loss_derivative(margin)) * label
        shrink = 1.0 - step * (2.0 * regularization)
        _finite_result(shrink, "kernel shrinkage factor")
        new_coefficients = shrink * coefficients
        new_coefficients[index] -= step * q
        new_intercept = intercept - step * q
    _finite_result(new_coefficients, "kernel coefficient update")
    _finite_result(new_intercept, "kernel intercept update")
    return new_coefficients, float(new_intercept)


def linear_batch_step(
    w: ArrayLike, b: float, X_batch: ArrayLike, y_batch: ArrayLike, eta: float
) -> tuple[NDArray[np.float64], float]:
    """One simultaneous minibatch step for the constrained linear objective.

    All margins use the supplied old ``w`` and ``b``. Average the sampled loss
    gradients, project the slope once, and leave the scalar intercept free.
    Repeated observations contribute once per occurrence to that average.
    """
    weights = _finite_array(w, "w", ndim=1)
    features = _finite_array(X_batch, "X_batch", ndim=2)
    labels = _finite_array(y_batch, "y_batch", ndim=1)
    if features.shape[0] == 0:
        raise ValueError("X_batch and y_batch must contain a nonempty batch")
    if features.shape != (labels.size, weights.size):
        raise ValueError("X_batch must match y_batch length and w feature count")
    if not np.all((labels == -1.0) | (labels == 1.0)):
        raise ValueError("y_batch must contain only -1 or +1")
    intercept = _finite_scalar(b, "b")
    step = _step_size(eta)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        margins = labels * (features @ weights + intercept)
        _finite_result(margins, "linear batch margins")
        q = dwd_loss_derivative(margins) * labels
        mean_gradient = (q @ features) / labels.size
        _finite_result(mean_gradient, "linear batch mean gradient")
        proposal = weights - step * mean_gradient
        new_intercept = intercept - step * float(np.mean(q))
    _finite_result(proposal, "linear batch weight update")
    _finite_result(new_intercept, "linear batch intercept update")
    return project_unit_ball(proposal), float(new_intercept)


def kernel_batch_step(
    a: ArrayLike,
    b: float,
    scores: ArrayLike,
    indices: ArrayLike,
    y_batch: ArrayLike,
    eta: float,
    lambd: float,
) -> tuple[NDArray[np.float64], float]:
    """One simultaneous RKHS minibatch step using precomputed old scores.

    ``scores`` contains complete old decision values, including ``b``, in batch
    order. It must be computed before changing either model parameter. Average
    the sampled loss increments over the actual batch length and apply the
    RKHS regularizer once, shrinking by ``1-2*eta*lambd`` for the objective
    mean DWD loss plus ``lambd * ||h||^2``. The finite-iterate norm bound
    additionally requires ``2*eta*lambd <= 1``.
    Repeated coefficient indices accumulate rather than overwrite each other.
    This function does not compute or cache kernels.
    """
    coefficients = _finite_array(a, "a", ndim=1)
    old_scores = _finite_array(scores, "scores", ndim=1)
    labels = _finite_array(y_batch, "y_batch", ndim=1)
    selected = np.asarray(indices)
    if selected.ndim != 1 or selected.dtype.kind not in "iu":
        raise ValueError("indices must be a one-dimensional array of integers")
    if selected.size == 0:
        raise ValueError("scores, indices, and y_batch must contain a nonempty batch")
    if old_scores.shape != selected.shape or labels.shape != selected.shape:
        raise ValueError("scores, indices, and y_batch must have the same length")
    if np.any(selected < 0) or np.any(selected >= coefficients.size):
        raise ValueError("indices must index existing kernel coefficients")
    if not np.all((labels == -1.0) | (labels == 1.0)):
        raise ValueError("y_batch must contain only -1 or +1")
    intercept = _finite_scalar(b, "b")
    step = _step_size(eta)
    regularization = _finite_scalar(lambd, "lambd")
    if regularization <= 0.0:
        raise ValueError("lambd must be strictly positive")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        margins = labels * old_scores
        _finite_result(margins, "kernel batch margins")
        q = dwd_loss_derivative(margins) * labels
        shrink = 1.0 - step * (2.0 * regularization)
        _finite_result(shrink, "kernel batch shrinkage factor")
        new_coefficients = shrink * coefficients
        np.add.at(new_coefficients, selected, -(step * q) / selected.size)
        new_intercept = intercept - step * float(np.mean(q))
    _finite_result(new_coefficients, "kernel batch coefficient update")
    _finite_result(new_intercept, "kernel batch intercept update")
    return new_coefficients, float(new_intercept)


def _rbf_recovery_arithmetic():
    """Qualify new recovery for nearest binary64 with retained subnormals.

    This observes caller arithmetic without changing its mode or NumPy policy.
    The predicate is shared in form with the independently checked affine
    readout gate. It is not a platform, library-vendor or version whitelist.
    """
    one = float(1)
    eps = float(np.finfo(float).eps)
    half, three_quarters = eps * .5, eps * .75
    tiny = float(np.finfo(float).smallest_subnormal)
    return (one + half == one and -one - half == -one
            and one + three_quarters == one + eps
            and -one - three_quarters == -one - eps
            and tiny * one == tiny and tiny != 0.0)


def _rbf_zero_distances_are_exact(values):
    """Certify duplicate-only zeros for finite binary64 direct distances.

    Allowed coordinates are zero or at least 2**-458 in magnitude. Unequal
    allowed values differ by at least 2**-510, whose square is normal. This
    certificate exempts only zero distances; positive tiny values still use
    the existing range recovery. Callers hoist it over shared query/basis data.
    """
    if type(values) is not np.ndarray or values.dtype != np.float64:
        return False
    # Flush modes can compare represented subnormal coordinates equal to zero.
    # Do not issue a represented-coordinate certificate in that arithmetic.
    if not _rbf_recovery_arithmetic():
        return False
    magnitude = np.abs(values)
    return bool(np.all((magnitude == 0.0) |
                       (np.isfinite(magnitude) & (magnitude >= float.fromhex("0x1p-458")))))


def _scale_rbf_distances(distances, left, right, width, *, trusted, contrast=False,
                         zero_distances_exact=False):
    """Preserve ordinary exponents; recover exceptional represented ranges.

    Only a captured built-in cdist result is eligible. Replaced helpers keep
    strict historical finite-distance and multiplication behavior. Inputs and
    width are already finite, real and shape-validated by the caller.
    """
    finite = bool(np.isfinite(distances).all())
    if not trusted:
        if not finite:
            raise FloatingPointError("squared Euclidean distances produced a nonfinite value")
        distances *= -width
        return
    info = np.finfo(np.float64)
    small_distance = info.tiny * left.shape[1]
    # If even a normal-range distance this small gives an exponent far below
    # epsilon, its lost subnormal terms cannot affect the rounded exp near one.
    # Existing expm1 consumers need small representable contrasts even when
    # exp itself would round to one. Ordinary Gaussian entries keep their gate.
    inspect_small = contrast or width > (info.eps / 8.0) / small_distance
    too_large = (width > 1.0 and finite and
                 float(np.max(distances, initial=0.0)) > info.max / width)
    small_entries = (distances <= small_distance) if inspect_small else None
    if contrast and zero_distances_exact:
        # The caller certified both operands once. Positive tiny entries,
        # including invalid negative values, retain their original handling.
        small_entries &= distances != 0.0
    tiny_present = inspect_small and bool(np.any(small_entries))
    if finite and not too_large and not tiny_present:
        distances *= -width
        return
    # Charge the arithmetic qualification only when new recovery is needed.
    # Nondefault modes retain the historical finite check and multiplication;
    # in particular, this does not repair existing directed-mode saturation.
    if not _rbf_recovery_arithmetic():
        if not finite:
            raise FloatingPointError("squared Euclidean distances produced a nonfinite value")
        distances *= -width
        return
    if (contrast and finite and not too_large and tiny_present
            and not np.any(distances)
            and np.all(left == right[0]) and np.all(right == right[0])):
        # A genuinely constant block has exactly zero pair distances. Do not
        # visit every pair with Python rational arithmetic in this common case.
        distances *= -width
        return
    if np.isnan(distances).any() or np.any(distances < 0.0):
        raise FloatingPointError("squared Euclidean distances produced an invalid value")
    repair = ~np.isfinite(distances)
    if width > 1.0:
        repair |= distances > info.max / width
    if inspect_small:
        repair |= small_entries
    if contrast:
        # Zero computed distances can denote exact duplicates or underflow.
        # Filter actual duplicates rowwise before entering the rational loop;
        # comparisons do not change their historical signed-zero arithmetic.
        zeros = repair & (distances == 0.0)
        for i in np.flatnonzero(np.any(zeros, axis=1)):
            columns = np.flatnonzero(zeros[i])
            identical = np.all(right[columns] == left[i], axis=1)
            repair[i, columns[identical]] = False
        if not np.any(repair):
            distances *= -width
            return
    # Save only exceptional coordinates; avoid overflowing ordinary multiply.
    pairs = tuple(zip(*np.nonzero(repair)))
    distances[repair] = 0.0
    distances *= -width
    from fractions import Fraction
    gamma = Fraction.from_float(float(width))
    for i, j in pairs:
        squared = Fraction(0)
        for x, y in zip(left[i], right[j]):
            if x != y:
                delta = Fraction.from_float(float(x)) - Fraction.from_float(float(y))
                squared += delta * delta
        try:
            exponent = float(gamma * squared)
        except OverflowError:
            exponent = math.inf
        distances[i, j] = -exponent


def rbf_kernel(X: ArrayLike, Y: ArrayLike, gamma: float) -> NDArray[np.float64]:
    """Return an RBF block ``exp(-gamma * ||X[i]-Y[j]||**2)``.

    This function retains no cache. Callers bound block sizes to control memory.
    Direct squared differences avoid cancellation from the norm/dot-product
    distance identity for points with large common offsets.
    """
    left = _finite_array(X, "X", ndim=2)
    right = _finite_array(Y, "Y", ndim=2)
    if left.shape[1] != right.shape[1] or left.shape[1] == 0:
        raise ValueError("X and Y must have the same positive feature count")
    width = _finite_scalar(gamma, "gamma")
    if width <= 0.0:
        raise ValueError("gamma must be strictly positive")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        distances = cdist(left, right, metric="sqeuclidean")
        _scale_rbf_distances(distances, left, right, width,
                             trusted=cdist is _ORIGINAL_CDIST)
        result = np.exp(distances, out=distances)
    _finite_result(result, "RBF kernel")
    return result


def select_near_boundary(
    scores: ArrayLike,
    fraction: float,
    rng: np.random.Generator,
    observation_ids: ArrayLike | None = None,
) -> NDArray[np.intp]:
    """Select ``floor(fraction * n)`` distinct rows by global absolute score.

    Return rows in rank order. Labels and score signs do not define quotas.
    Independent random keys break score ties, then observation IDs break key
    ties. Callers should provide unique IDs for deterministic final tie breaks.
    Empty scores or a count rounded to zero yield empty indices; class and
    fit-size checks belong to the estimator because this operation does not
    inspect labels.
    """
    values = _finite_array(scores, "scores", ndim=1)
    retained_fraction = _finite_scalar(fraction, "fraction")
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    if observation_ids is None:
        ids = np.arange(values.size, dtype=np.intp)
    else:
        ids = np.asarray(observation_ids)
        if ids.ndim != 1 or ids.size != values.size:
            raise ValueError("observation_ids must be a vector matching scores")
        if ids.dtype.kind == "c":
            raise ValueError("observation_ids must be real numbers or comparable identifiers")
        if ids.dtype.kind == "f" and not np.all(np.isfinite(ids)):
            raise ValueError("observation_ids must be finite")
        try:
            if np.unique(ids).size != ids.size:
                raise ValueError("observation_ids must be unique")
        except TypeError as exc:
            raise ValueError("observation_ids must have mutually comparable values") from exc
    count = math.floor(retained_fraction * values.size)
    keys = rng.random(values.size)
    try:
        order = np.lexsort((ids, keys, np.abs(values)))
    except TypeError as exc:
        raise ValueError("observation_ids must have mutually comparable values") from exc
    return order[:count]
