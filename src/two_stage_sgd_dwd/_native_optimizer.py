"""Optional native scalar updates with the original Python sampling schedule.

Only one epoch of draws and step sizes is prepared at a time. This module
never prepares a Gram matrix and never changes a native failure into a Python
retry. The default estimator backend remains Python.
"""
from __future__ import annotations

import copy
import importlib
import time

import numpy as np

from ._kernels import _TRUSTED


def resolve_backend(kind, X, validation, est, deadline, callback):
    """Return the optional extension, recording explicit selection diagnostics."""
    requested = est.optimizer_backend
    reason = None
    if requested == "python":
        reason = "requested_python"
    elif est.batch_size != 1:
        reason = "batch_size_not_one"
    elif deadline is not None:
        reason = "deadline"
    elif callback is not None:
        reason = "checkpoint_callback"
    elif validation is not None:
        reason = "validation_data"
    elif not X.flags.c_contiguous or not X.flags.aligned:
        # Copying strided rows could change NumPy dot's reduction layout.
        reason = "noncontiguous_or_unaligned_features"
    elif kind == "kernel" and (not isinstance(est.kernel, str) or est.kernel not in _TRUSTED):
        reason = "unsupported_kernel"
    if reason is not None:
        est.optimizer_backend_ = "python"
        est.optimizer_backend_reason_ = reason
        if requested == "native":
            est.optimizer_backend_ = "unrun"
            raise ValueError(f"optimizer_backend='native' does not support {reason}")
        return None
    try:
        module = importlib.import_module("._sgd_accel", __package__)
    except (ImportError, OSError) as exc:
        est.optimizer_backend_reason_ = "extension_unavailable"
        est.optimizer_backend_ = "python" if requested == "auto" else "unrun"
        if requested == "native":
            raise ImportError("optimizer_backend='native' requires the optional compiled _sgd_accel extension") from exc
        return None
    est.optimizer_backend_, est.optimizer_backend_reason_ = "native", "native_supported"
    return module


def optimize(kind, X, y, rng, est, native, column_cache=None):
    """Optimize supported fixed-epoch fits; preserve Python's committed prefix."""
    # Runtime import avoids a module cycle and retains the estimator's shared
    # snapshot and column-evaluation functions as the authoritative paths.
    from . import _estimators as helpers

    began = time.perf_counter()
    m = len(y)
    coefficients = np.zeros(X.shape[1] if kind == "linear" else m)
    intercept = 0.0
    if kind == "linear":
        with np.errstate(over="raise", invalid="raise"):
            B = 4.0 * float(np.max(np.sum(X * X, axis=1) + 1.0))
    else:
        B = 2.0 * est.lambd_ + 4.0 * (est.max_kernel_diagonal_ + 1.0)
    if not np.isfinite(B) or B <= 0:
        raise helpers.NumericalFailure("The learning-rate bound is nonfinite")
    initial = helpers._snapshot(coefficients, intercept, 0, rng)
    kernel_counter = [0]
    if kind == "linear":
        rows = tuple(X[i] for i in range(m))
        get_row = rows.__getitem__
    else:
        def get_row(i):
            if column_cache is None:
                values = helpers._kernel_values(X, X[i:i + 1], est.gamma_, est._kernel_spec_)[:, 0]
                kernel_counter[0] += m
                return values
            values, hit = column_cache.get(i, lambda: helpers._kernel_values(
                X, X[i:i + 1], est.gamma_, est._kernel_spec_)[:, 0])
            if not hit:
                kernel_counter[0] += m
            return values
    updates = 0
    update_seconds = 0.0
    error = None
    for _ in range(int(est.max_epochs)):
        began_update = time.perf_counter()
        try:
            epoch_rng = copy.deepcopy(rng.bit_generator.state)
            if est.sampling == "shuffle":
                indices = np.asarray(rng.permutation(m), dtype=np.int64)
            else:
                # A vectorized draw is not assumed to preserve the original
                # scalar RNG stream for every NumPy generator implementation.
                indices = np.asarray([int(rng.integers(m)) for _ in range(m)], dtype=np.int64)
            rates = np.asarray([(1.0 / B) * (1.0 + (updates + j) / m) ** (-0.75)
                                for j in range(m)], dtype=np.float64)
            coefficients, intercept, completed, error = native.run_steps(
                0 if kind == "linear" else 1, coefficients, intercept, get_row,
                y, indices, rates, 0.0 if kind == "linear" else est.lambd_)
            if error is not None:
                # Python consumes the failed observation's draw but commits
                # neither its coefficient update nor its sample count.
                rng.bit_generator.state = epoch_rng
                if est.sampling == "shuffle":
                    rng.permutation(m)
                else:
                    for _ in range(completed + 1):
                        rng.integers(m)
            updates += completed
        finally:
            update_seconds += time.perf_counter() - began_update
        if error is not None:
            break
    final = helpers._snapshot(coefficients, intercept, updates, rng, updates)
    selected = copy.deepcopy(final)
    history = [{"iteration": 0, "epoch": 0.0, "n_samples_seen": 0,
                "validation_loss": None, "elapsed_seconds": 0.0,
                "selected": updates == 0, "patience_reference_loss": None,
                "patience_misses": 0}]
    return {"initial_state_": initial, "final_state_": final, "selected_state_": selected,
            "selected_iteration_": updates, "n_updates_": updates,
            "selected_n_samples_seen_": updates, "n_samples_seen_": updates,
            "n_epochs_": updates / m, "batch_size_": 1, "n_updates_per_epoch_": m,
            "sampling_": est.sampling, "learning_rate_clock_": "processed_observations",
            "learning_rate_B_": B, "history_": history,
            "stop_reason_": "numerical_failure" if error is not None else "epoch_cap",
            "model_status_": "numerical_failure" if error is not None else "trained",
            "validation_available_": False, "best_validation_loss_": None,
            "update_seconds_": update_seconds, "validation_checkpoint_seconds_": 0.0,
            "optimization_seconds_": time.perf_counter() - began,
            "kernel_evaluation_count_": kernel_counter[0], "error_": error,
            "rng_state_": copy.deepcopy(rng.bit_generator.state)}
