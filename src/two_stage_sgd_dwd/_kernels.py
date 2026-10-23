"""Resolved on-demand kernel specifications compatible with ensemble DWD.

Built-in PSD families need only analytic training diagonals. Sigmoid, sklearn's
negative additive-chi-square kernel, and custom matrix callables instead receive
a one-time full training-Gram symmetry/PSD check in ``prepare``. That matrix is
transient and discarded before SGD; it is not a persistent kernel cache.

A finite training check cannot certify a custom kernel on its entire domain.
Callers remain responsible for providing a symmetric PSD kernel on all inputs.
Custom callables must be pickleable themselves if their KernelSpec is serialized.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import numbers
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics.pairwise import pairwise_kernels

from ._core import _contains_complex, _finite_array, _finite_result, rbf_kernel


_TRUSTED = frozenset(("linear", "rbf", "poly", "polynomial", "laplacian", "cosine", "chi2"))
_NAMED = _TRUSTED | {"sigmoid", "additive_chi2"}
_USES_GAMMA = frozenset(("rbf", "poly", "polynomial", "laplacian", "chi2", "sigmoid"))
_KEYWORDS = {
    "linear": {"dense_output"},
    "rbf": set(),
    "poly": {"degree", "coef0"},
    "polynomial": {"degree", "coef0"},
    "laplacian": set(),
    "cosine": {"dense_output"},
    "chi2": set(),
    "sigmoid": {"coef0"},
    "additive_chi2": set(),
}


def _matrix(value: ArrayLike, name: str, *, allow_empty: bool = False) -> NDArray[np.float64]:
    result = _finite_array(value, name, ndim=2)
    if result.shape[1] == 0 or (result.shape[0] == 0 and not allow_empty):
        raise ValueError(f"{name} must have a positive feature count and at least one row")
    return result


def _real(value, name, *, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not np.isfinite(value)):
        raise ValueError(f"{name} must be a finite real number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _check_domain(kernel, X):
    if isinstance(kernel, str) and kernel in ("chi2", "additive_chi2") and np.any(X < 0):
        raise ValueError(f"{kernel} requires nonnegative features")


def _unit_rows(X):
    # Scaling each row first prevents overflow in norms of large finite inputs.
    scale = np.max(np.abs(X), axis=1, keepdims=True)
    scaled = np.divide(X, scale, out=np.zeros_like(X), where=scale != 0)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm != 0)


@dataclass(frozen=True)
class KernelSpec:
    """A resolved kernel with independent keyword storage and no retained Gram.

    ``gamma`` is None for kernels without a built-in gamma parameter and for
    callables. Callable-specific gamma must be supplied explicitly in ``kws``.
    Keyword arrays are copied during resolution; callers should treat this fitted
    specification as read-only. A callable receives two feature matrices and
    returns a matrix with one row per left input and one column per right input.
    """

    kernel: str | Callable
    gamma: float | None
    kws: dict

    def evaluate(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Compute one finite left-by-right kernel block without input mutation."""
        left = _matrix(X, "X", allow_empty=True)
        right = _matrix(Y, "Y", allow_empty=True)
        if left.shape[1] != right.shape[1]:
            raise ValueError("X and Y must have the same feature count")
        _check_domain(self.kernel, left)
        _check_domain(self.kernel, right)
        shape = (len(left), len(right))
        if 0 in shape:
            return np.empty(shape, dtype=np.float64)
        owned_output = False
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if callable(self.kernel):
                # A user function receives private arrays and keywords: mutations
                # cannot alter fitting rows, future predictions, or the spec.
                values = self.kernel(left.copy(), right.copy(), **deepcopy(self.kws))
                supplied = np.asarray(values)
                if _contains_complex(supplied):
                    raise ValueError("callable kernel output must be real")
                try:
                    values = np.asarray(supplied, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError("callable kernel output must be a real matrix") from exc
                if values.ndim != 2:
                    raise ValueError("callable kernel output must be a two-dimensional matrix")
            elif self.kernel == "rbf":
                values = (np.ones(shape) if self.gamma == 0.0
                          else rbf_kernel(left, right, self.gamma))
                owned_output = self.gamma == 0.0
            elif self.kernel == "cosine":
                left_unit, right_unit = _unit_rows(left), _unit_rows(right)
                values = left_unit @ right_unit.T
                owned_output = type(left_unit) is np.ndarray and type(right_unit) is np.ndarray
                del left_unit, right_unit
            elif self.kernel in ("poly", "polynomial") and self.kws["degree"] == 0:
                values = np.ones(shape)
                owned_output = True
            elif self.gamma == 0.0:
                if self.kernel in ("laplacian", "chi2"):
                    constant = 1.0
                elif self.kernel in ("poly", "polynomial"):
                    constant = self.kws["coef0"] ** self.kws["degree"]
                elif self.kernel == "sigmoid":
                    constant = np.tanh(self.kws["coef0"])
                else:  # Resolution never assigns gamma to other named kernels.
                    raise ValueError("Unexpected gamma for this kernel")
                values = np.full(shape, constant, dtype=np.float64)
                owned_output = True
            else:
                values = pairwise_kernels(left, right, metric=self.kernel, **self.kws)
        if values.shape != shape:
            raise ValueError(f"Kernel output must have shape {shape}, got {values.shape}")
        _finite_result(values, "kernel block")
        # Only locally allocated outputs have no aliases retained by a producer.
        if (owned_output and type(values) is np.ndarray and values.dtype == np.float64
                and values.flags.owndata and values.flags.writeable):
            return values
        return np.array(values, dtype=np.float64, copy=True)

    def prepare(self, X: ArrayLike, block_size: int = 256, check_deadline=None) -> dict:
        """Validate training geometry and return the diagonal bound for SGD.

        Trusted-family evaluation counts include one analytically evaluated
        diagonal per row; no pairwise matrix is constructed. Other kernels use
        m*m evaluated values plus a transient full Gram matrix/eigendecomposition.
        This validates only the observed training matrix, not global PSD.
        Call this once on the actual fitting set before optimization.

        On failure, ``evaluation_count`` attached to the exception records values
        successfully returned or analytic diagonals evaluated before interruption.
        Work inside a failed kernel call or eigensolve cannot be counted.
        """
        progress = {"evaluation_count": 0}
        try:
            return self._prepare(X, block_size, check_deadline, progress)
        except Exception as exc:
            try:
                exc.evaluation_count = progress["evaluation_count"]
            except (AttributeError, TypeError):
                pass
            raise

    def _prepare(self, X, block_size, check_deadline, progress):
        if (isinstance(block_size, (bool, np.bool_)) or
                not isinstance(block_size, numbers.Integral) or block_size < 1):
            raise ValueError("block_size must be a positive integer")
        if check_deadline is not None and not callable(check_deadline):
            raise ValueError("check_deadline must be callable or None")

        def check():
            if check_deadline is not None:
                check_deadline()

        check()
        rows = _matrix(X, "X")
        _check_domain(self.kernel, rows)
        m, block_size = len(rows), int(block_size)
        trusted = isinstance(self.kernel, str) and self.kernel in _TRUSTED
        if trusted:
            maximum = 0.0
            for start in range(0, m, block_size):
                check()
                block = rows[start:start + block_size]
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    if self.kernel in ("rbf", "laplacian", "chi2"):
                        diagonal = np.ones(len(block))
                    elif self.kernel == "cosine":
                        diagonal = np.any(block != 0.0, axis=1).astype(float)
                    elif self.kernel in ("poly", "polynomial") and self.kws["degree"] == 0:
                        diagonal = np.ones(len(block))
                    elif self.kernel in ("poly", "polynomial") and self.gamma == 0.0:
                        diagonal = np.full(len(block), self.kws["coef0"] ** self.kws["degree"])
                    else:
                        diagonal = np.sum(block * block, axis=1)
                        if self.kernel in ("poly", "polynomial"):
                            diagonal = (self.gamma * diagonal + self.kws["coef0"]) ** self.kws["degree"]
                progress["evaluation_count"] += len(diagonal)
                _finite_result(diagonal, "kernel diagonal")
                maximum = max(maximum, float(np.max(diagonal)))
                check()
            return {"max_diagonal": maximum, "evaluation_count": m,
                    "validation": "analytic_psd_family"}

        check()
        gram = np.empty((m, m), dtype=np.float64)
        count = 0
        scale = 0.0
        for start in range(0, m, block_size):
            check()
            block = self.evaluate(rows[start:start + block_size], rows)
            gram[start:start + len(block)] = block
            count += block.size
            progress["evaluation_count"] = count
            scale = max(scale, float(np.max(np.abs(block))))
            check()
        diagonal = np.diag(gram)
        if np.any(diagonal < 0.0):
            raise ValueError("A PSD training kernel must have a nonnegative diagonal")
        maximum = float(np.max(diagonal))
        # Scale only this private validation matrix. Checking in normalized units
        # makes the verdict independent of a positive kernel multiplier, and
        # avoids underflow of relative tolerances for subnormal kernel values.
        # Divide directly rather than multiplying by a potentially infinite 1/scale.
        if scale > 0.0:
            for start in range(0, m, block_size):
                check()
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    gram[start:start + block_size] /= scale
                check()
        tolerance = 100 * np.finfo(float).eps * m if scale > 0.0 else 0.0
        _finite_result(tolerance, "kernel symmetry tolerance")
        for start in range(0, m, block_size):
            check()
            stop = min(start + block_size, m)
            with np.errstate(over="raise", invalid="raise"):
                discrepancy = float(np.max(np.abs(gram[start:stop] - gram[:, start:stop].T)))
            if discrepancy > tolerance:
                raise ValueError("The training kernel must be symmetric")
            check()
        check()
        try:
            eigenvalues = np.linalg.eigvalsh(gram)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Could not validate the training kernel's positive semidefiniteness") from exc
        check()
        _finite_result(eigenvalues, "training kernel eigenvalues")
        psd_tolerance = 100 * np.finfo(float).eps * m * float(np.max(np.abs(eigenvalues)))
        _finite_result(psd_tolerance, "kernel PSD tolerance")
        if float(np.min(eigenvalues)) < -psd_tolerance:
            raise ValueError("The training kernel must be positive semidefinite")
        # The evaluated kernel is never clipped, symmetrized, or shifted. Only
        # the disposable validation matrix was scaled; no Gram is retained.
        del gram, eigenvalues, diagonal, block
        return {"max_diagonal": maximum, "evaluation_count": count,
                "validation": "training_gram_psd_check"}


def resolve_kernel(
    X: ArrayLike, kernel="rbf", gamma=1.0, kernel_kws=None
) -> KernelSpec:
    """Freeze built-in width/keywords using fitting features, never query rows.

    Gamma follows ensemble DWD: ``'auto'`` is 1/d; ``'scale'`` is 1/(d*X.var())
    or 1 for constant features; finite nonnegative numeric gamma is accepted.
    Built-in gamma belongs only at the top level. Callable keyword arguments,
    including any custom gamma, are passed only through ``kernel_kws``.
    """
    rows = _matrix(X, "X")
    if not isinstance(kernel, str) and not callable(kernel):
        raise ValueError("kernel must be a supported name or a matrix-level callable")
    if isinstance(kernel, str) and kernel not in _NAMED:
        if kernel == "precomputed":
            raise ValueError("Precomputed kernels are unsupported; supply feature rows")
        raise ValueError(f"Unsupported kernel name: {kernel!r}")
    if kernel_kws is not None and not isinstance(kernel_kws, dict):
        raise ValueError("kernel_kws must be a dictionary or None")
    kws = deepcopy(kernel_kws) if kernel_kws is not None else {}
    if callable(kernel):
        return KernelSpec(kernel, None, kws)
    if isinstance(gamma, str):
        if gamma not in ("scale", "auto"):
            raise ValueError("gamma must be 'scale', 'auto', or a finite nonnegative number")
    else:
        gamma = _real(gamma, "gamma", nonnegative=True)
    if "gamma" in kws:
        raise ValueError("Set built-in gamma with the top-level gamma parameter, not kernel_kws")
    unsupported = set(kws) - _KEYWORDS[kernel]
    if unsupported:
        raise ValueError(f"Unsupported {kernel} kernel keywords: {sorted(unsupported, key=str)}")
    if "dense_output" in kws and not isinstance(kws["dense_output"], (bool, np.bool_)):
        raise ValueError("dense_output must be boolean")
    _check_domain(kernel, rows)
    if kernel in ("poly", "polynomial"):
        degree = kws.get("degree", 3)
        if (isinstance(degree, (bool, np.bool_)) or
                not isinstance(degree, numbers.Integral) or degree < 0):
            raise ValueError("Polynomial degree must be a nonnegative integer for a PSD kernel")
        kws["degree"] = int(degree)
        kws["coef0"] = _real(kws.get("coef0", 1.0), "coef0", nonnegative=True)
    elif kernel == "sigmoid":
        kws["coef0"] = _real(kws.get("coef0", 1.0), "coef0")
    resolved_gamma = None
    if kernel in _USES_GAMMA:
        if gamma == "auto":
            resolved_gamma = 1.0 / rows.shape[1]
        elif gamma == "scale":
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                variance = float(rows.var())
            if not np.isfinite(variance):
                raise ValueError("Feature variance overflowed; rescale X or specify numeric gamma")
            if variance == 0.0 and rows.min() != rows.max():
                raise ValueError("Feature variance underflowed; rescale X or specify numeric gamma")
            resolved_gamma = (1.0 / rows.shape[1]) / variance if variance != 0.0 else 1.0
            if not np.isfinite(resolved_gamma):
                raise ValueError("Scaled gamma overflowed; rescale X or specify numeric gamma")
            if resolved_gamma == 0.0:
                raise ValueError("Scaled gamma underflowed; rescale X or specify numeric gamma")
        else:
            resolved_gamma = float(gamma)
        kws["gamma"] = resolved_gamma
    return KernelSpec(kernel, resolved_gamma, kws)
