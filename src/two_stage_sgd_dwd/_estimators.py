"""Binary estimators for the Chapter 3 method; no external DWD optimizer."""
from __future__ import annotations

import copy
import numbers
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from ._core import (dwd_loss, kernel_batch_step, kernel_step, linear_batch_step,
                    linear_step, rbf_kernel, select_near_boundary)
from ._cache import LazyKernelColumnCache
from ._kernels import _TRUSTED, resolve_kernel
from ._prediction import rbf_prediction_scores


class TrainingError(RuntimeError):
    """A fit failed; ``status`` identifies its scientific failure category."""
    def __init__(self, message, status="training_failure"):
        super().__init__(message)
        self.status = status


class NumericalFailure(TrainingError):
    """A nonfinite calculation invalidated the fit, including older checkpoints."""
    def __init__(self, message, status="numerical_failure"):
        super().__init__(message, status)


class SelectionError(ValueError):
    """The selected observations do not contain both training classes."""
    def __init__(self, message, status="single_class_selection"):
        super().__init__(message)
        self.status = status


class _Deadline(Exception):
    pass


def _generator(random_state):
    if isinstance(random_state, np.random.Generator):
        return copy.deepcopy(random_state)
    if isinstance(random_state, np.random.RandomState):
        state = copy.deepcopy(random_state)
        return np.random.default_rng(int(state.randint(0, 2**32, dtype=np.uint64)))
    if random_state is not None and (isinstance(random_state, bool) or
            not isinstance(random_state, numbers.Integral) or random_state < 0):
        raise ValueError("random_state must be a nonnegative integer, NumPy RNG, or None")
    return np.random.default_rng(random_state)


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _penalty(est):
    """Freeze the coefficient of the full squared RKHS norm, as in dwd."""
    _positive(est.lambd, "lambd")
    est.lambd_ = float(est.lambd)


def _kernel_values(X, Y, gamma, spec=None):
    # Keep the original RBF path and its arithmetic for default-fit compatibility.
    if spec is None or (spec.kernel == "rbf" and gamma > 0):
        return rbf_kernel(X, Y, gamma)
    return spec.evaluate(X, Y)


def _set_kernel(est, X):
    est._kernel_input_kws_ = copy.deepcopy(est.kernel_kws)
    spec = resolve_kernel(X, est.kernel, est.gamma, est._kernel_input_kws_)
    est._kernel_spec_ = spec
    est.kernel_ = spec.kernel
    est.gamma_ = spec.gamma
    est.kernel_kws_ = copy.deepcopy(spec.kws)
    return spec


def _controls(est):
    for name in ("max_epochs", "validation_interval", "patience"):
        _integer(getattr(est, name), name)
    _integer(est.min_epochs, "min_epochs", 0)
    _positive(est.tol, "tol", True)
    if not (isinstance(est.batch_size, str) and est.batch_size == "full"):
        _integer(est.batch_size, "batch_size (positive integer or 'full')")
    if not isinstance(est.sampling, str) or est.sampling not in ("shuffle", "with_replacement"):
        raise ValueError("sampling must be 'shuffle' or 'with_replacement'")
    if not isinstance(est.stopping, str) or est.stopping not in ("auto", "fixed", "validation"):
        raise ValueError("stopping must be 'auto', 'fixed', or 'validation'")
    if not isinstance(est.optimizer_backend, str) or est.optimizer_backend not in ("python", "auto", "native"):
        raise ValueError("optimizer_backend must be 'python', 'auto', or 'native'")
    est.optimizer_backend_, est.optimizer_backend_reason_ = "unrun", "not_started"
    if est.max_seconds is not None:
        _positive(est.max_seconds, "max_seconds")
    _generator(est.random_state)


_CACHE_DIAGNOSTICS = ("kernel_cache_hits_", "kernel_cache_misses_",
                      "kernel_cache_evictions_", "kernel_cache_peak_bytes_",
                      "kernel_cache_capacity_columns_")


def _cache_controls(est):
    _integer(est.kernel_cache_bytes, "kernel_cache_bytes", 0)
    for name in _CACHE_DIAGNOSTICS:
        setattr(est, name, 0)
    if est.kernel_cache_bytes:
        if est.batch_size != 1:
            raise ValueError("Positive kernel_cache_bytes requires batch_size=1")
        if not isinstance(est.kernel, str) or est.kernel not in _TRUSTED:
            raise ValueError("Positive kernel_cache_bytes requires an analytic built-in kernel: "
                             "linear, rbf, poly, polynomial, laplacian, cosine, or chi2")


def _resolve_stopping(est, validation):
    if est.stopping == "validation" and validation is None:
        raise ValueError("stopping='validation' requires explicit validation_data")
    est.stopping_ = ("validation" if validation is not None else "fixed") if est.stopping == "auto" else est.stopping


def _fit_diagnostics(est):
    """Describe usability and stopping without inferring SGD optimality.

    n_iter_ counts optimizer updates, not MM iterations or epochs. Completing a
    fixed epoch budget meets only that budget criterion. No stationarity or
    optimality certificate is computed by this solver.
    """
    reason = est.stop_reason_
    est.n_iter_ = int(getattr(est, "n_updates_", 0))
    est.returned_iteration_ = int(getattr(est, "selected_iteration_", 0))
    est.n_epochs_completed_ = float(getattr(est, "n_epochs_", 0.0))
    est.termination_reason_ = reason
    est.termination_criterion_ = "validation_patience" if est.stopping_ == "validation" else "fixed_epochs"
    est.iteration_limit_reached_ = est.n_epochs_completed_ >= est.max_epochs
    est.time_limit_reached_ = reason in ("time_cap", "time_cap_no_validation")
    est.cap_reached_ = est.iteration_limit_reached_ or est.time_limit_reached_
    est.fit_usable_ = bool(getattr(est, "_fitted_", False) and est.model_status_ == "trained")
    est.criterion_reached_ = bool(est.fit_usable_ and (
        reason == "validation_patience" if est.stopping_ == "validation"
        else est.iteration_limit_reached_))
    est.checkpoint_selection_ = "validation_loss" if getattr(est, "validation_available_", False) else "final_iterate"
    est.optimality_checked_ = False
    est.optimality_met_ = False
    est.converged_ = False
    # A finite epoch-limited model remains a valid fixed-budget CV candidate.
    # Wall-time truncation, no-update fallbacks and failed fits do not.
    est.cv_eligible_ = bool(est.fit_usable_ and not est.time_limit_reached_)


_STAGE_DIAGNOSTICS = (
    "n_iter_", "returned_iteration_", "n_epochs_completed_", "stopping_",
    "termination_reason_", "termination_criterion_", "criterion_reached_",
    "iteration_limit_reached_", "time_limit_reached_", "cap_reached_",
    "fit_usable_", "cv_eligible_", "checkpoint_selection_",
    "optimality_checked_", "optimality_met_", "converged_", "model_status_",
    "optimizer_backend_", "optimizer_backend_reason_",
) + _CACHE_DIAGNOSTICS


def _two_stage_diagnostics(est):
    """Keep both stages visible, including partial and failed pipeline fits."""
    models = [getattr(est, name, None) for name in ("linear_estimator_", "kernel_estimator_")]
    est.stage_diagnostics_ = {
        stage: None if model is None else {
            name[:-1]: copy.deepcopy(getattr(model, name))
            for name in _STAGE_DIAGNOSTICS if hasattr(model, name)
        }
        for stage, model in zip(("linear", "kernel"), models)
    }
    for name, default, dtype in (("n_iter_", 0, np.int64), ("returned_iteration_", 0, np.int64),
                                 ("n_epochs_completed_", 0.0, float),
                                 ("converged_", False, bool), ("optimality_met_", False, bool),
                                 ("optimality_checked_", False, bool)):
        setattr(est, name, np.asarray([getattr(model, name, default) for model in models], dtype=dtype))
    est.termination_reason_ = est.stop_reason_
    est.termination_criterion_ = "validation_patience" if est.stopping_ == "validation" else "fixed_epochs"
    est.fit_usable_ = bool(getattr(est, "_fitted_", False) and
                          all(getattr(model, "fit_usable_", False) for model in models))
    est.cv_eligible_ = bool(est.fit_usable_ and
                           all(getattr(model, "cv_eligible_", False) for model in models))
    est.criterion_reached_ = bool(est.fit_usable_ and
                                 all(getattr(model, "criterion_reached_", False) for model in models))
    est.iteration_limit_reached_ = any(getattr(model, "iteration_limit_reached_", False) for model in models)
    est.time_limit_reached_ = (est.stop_reason_ in ("time_cap", "time_cap_no_validation") or
                               any(getattr(model, "time_limit_reached_", False) for model in models))
    est.cap_reached_ = est.iteration_limit_reached_ or est.time_limit_reached_
    for name in _CACHE_DIAGNOSTICS:
        setattr(est, name, getattr(models[1], name, 0))
    backends = [getattr(model, "optimizer_backend_", "unrun") for model in models]
    est.optimizer_backend_ = backends[0] if backends[0] == backends[1] else "mixed"
    est.optimizer_backend_reason_ = {
        stage: getattr(model, "optimizer_backend_reason_", "not_started")
        for stage, model in zip(("linear", "kernel"), models)}


def _clear(est):
    for name in list(vars(est)):
        if name.endswith("_") and not name.startswith("__"):
            delattr(est, name)


def _training_data(est, X, y, validation_data):
    X, y = validate_data(est, X, y, dtype=np.float64, accept_sparse=False,
                         ensure_min_samples=2)
    check_classification_targets(y)
    classes = unique_labels(y)
    if len(classes) != 2:
        raise ValueError("DWD requires exactly two classes")
    est.classes_ = classes.copy()
    encoded = np.where(y == classes[1], 1.0, -1.0)
    validation = None
    if validation_data is not None:
        if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
            raise ValueError("validation_data must be (X_validation, y_validation)")
        Xv = check_array(validation_data[0], dtype=np.float64, accept_sparse=False)
        yv = np.asarray(validation_data[1])
        if Xv.shape[1] != X.shape[1] or yv.ndim != 1 or len(yv) != len(Xv):
            raise ValueError("Validation dimensions do not match training data")
        check_classification_targets(yv)
        if not np.isin(yv, classes).all():
            raise ValueError("Validation labels must belong to the training classes")
        validation = (Xv, np.where(yv == classes[1], 1.0, -1.0))
    return X, y, encoded, validation


def _check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise _Deadline


def _scores(kind, X, coefficients, intercept, basis=None, gamma=None,
            batch_size=256, deadline=None, kernel_counter=None, kernel_spec=None):
    result = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        _check_deadline(deadline)
        block = X[start:start + batch_size]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if kind == "linear":
                values = block @ coefficients + intercept
            else:
                kernel = _kernel_values(block, basis, gamma, kernel_spec)
                if kernel_counter is not None:
                    kernel_counter[0] += len(block) * len(basis)
                values = kernel @ coefficients + intercept
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite decision scores")
        result[start:start + len(block)] = values
        _check_deadline(deadline)
    return result


def _snapshot(coefficients, intercept, iteration, rng, samples_seen=None):
    return {"coefficients": coefficients.copy(), "intercept": float(intercept),
            "iteration": int(iteration),
            "n_samples_seen": int(iteration if samples_seen is None else samples_seen),
            "rng_state": copy.deepcopy(rng.bit_generator.state)}


def _optimize(kind, X, y, validation, rng, est, deadline, callback=None):
    # The cache never becomes part of a fitted model or callback snapshot.
    # A finally block also releases entries on unexpected callback failures.
    cache = (LazyKernelColumnCache(est.kernel_cache_bytes, len(y))
             if kind == "kernel" and est.kernel_cache_bytes else None)
    try:
        from ._native_optimizer import optimize, resolve_backend
        native = resolve_backend(kind, X, validation, est, deadline, callback)
        result = (_optimize_steps(kind, X, y, validation, rng, est, deadline, callback, cache)
                  if native is None else optimize(kind, X, y, rng, est, native, cache))
        if cache is not None:
            result.update(cache.diagnostics())
        return result
    finally:
        if cache is not None:
            cache.clear()
            for name, value in cache.diagnostics().items():
                setattr(est, name, value)


def _optimize_steps(kind, X, y, validation, rng, est, deadline, callback=None,
                    column_cache=None):
    began = time.perf_counter()
    m = len(y)
    full_batch = isinstance(est.batch_size, str) and est.batch_size == "full"
    batch_size = m if full_batch else min(int(est.batch_size), m)
    steps_per_epoch = (m + batch_size - 1) // batch_size
    coefficients = np.zeros(X.shape[1] if kind == "linear" else m)
    intercept = 0.0
    if kind == "linear":
        with np.errstate(over="raise", invalid="raise"):
            B = 4.0 * float(np.max(np.sum(X * X, axis=1) + 1.0))
    else:
        B = 2.0 * est.lambd_ + 4.0 * (est.max_kernel_diagonal_ + 1.0)
    if not np.isfinite(B) or B <= 0:
        raise NumericalFailure("The learning-rate bound is nonfinite")
    initial = _snapshot(coefficients, intercept, 0, rng)
    history = [{"iteration": 0, "epoch": 0.0, "n_samples_seen": 0, "validation_loss": None,
                "elapsed_seconds": 0.0, "selected": False,
                "patience_reference_loss": None, "patience_misses": 0}]
    if callback is not None:
        callback(copy.deepcopy(initial), copy.deepcopy(history[0]))
    best, best_loss, reference, misses = None, np.inf, None, 0
    updates, samples_seen, updates_time, validation_time = 0, 0, 0.0, 0.0
    kernel_counter = [0]
    order = None
    stop, error = "epoch_cap", None
    for t in range(int(est.max_epochs) * steps_per_epoch):
        try:
            _check_deadline(deadline)
            began_update = time.perf_counter()
            try:
                # Preserve the original scalar draw and arithmetic for size one.
                # Every other batch evaluates all margins at this same old state.
                count = min(batch_size, m - samples_seen % m)
                eta = (1.0 / B) * (1.0 + samples_seen / m) ** (-0.75)
                if not full_batch and est.sampling == "shuffle":
                    position = samples_seen % m
                    if position == 0:
                        order = rng.permutation(m)
                    indices = order[position:position + count]
                elif full_batch:
                    indices = np.arange(m)
                elif batch_size != 1:
                    indices = rng.integers(m, size=count)
                if batch_size == 1:
                    i = int(indices[0]) if est.sampling == "shuffle" else int(rng.integers(m))
                    if kind == "linear":
                        new_coefficients, new_intercept = linear_step(coefficients, intercept,
                                                                      X[i], y[i], eta)
                    else:
                        if column_cache is None:
                            k = _kernel_values(X, X[i:i + 1], est.gamma_, est._kernel_spec_)[:, 0]
                            kernel_counter[0] += m
                        else:
                            k, hit = column_cache.get(i, lambda: _kernel_values(
                                X, X[i:i + 1], est.gamma_, est._kernel_spec_)[:, 0])
                            if not hit:
                                kernel_counter[0] += m
                        new_coefficients, new_intercept = kernel_step(
                            coefficients, intercept, k, i, y[i], eta, est.lambd_)
                else:
                    if kind == "linear":
                        new_coefficients, new_intercept = linear_batch_step(
                            coefficients, intercept, X[indices], y[indices], eta)
                    else:
                        # One kernel block at a time, including full-batch mode.
                        # A deadline during scoring leaves the entire batch uncommitted.
                        scores = _scores("kernel", X[indices], coefficients, intercept,
                                         X, est.gamma_, est.prediction_batch_size, deadline,
                                         kernel_counter=kernel_counter, kernel_spec=est._kernel_spec_)
                        new_coefficients, new_intercept = kernel_batch_step(
                            coefficients, intercept, scores, indices, y[indices],
                            eta, est.lambd_)
                # Commit only a wholly finite update. Older states remain diagnostic
                # if a later numerical failure occurs; they do not rescue that fit.
                if not np.isfinite(new_coefficients).all() or not np.isfinite(new_intercept):
                    raise FloatingPointError("Nonfinite SGD state")
                coefficients, intercept = new_coefficients, float(new_intercept)
                updates = t + 1
                samples_seen += count
            finally:
                updates_time += time.perf_counter() - began_update
            _check_deadline(deadline)
            if validation is not None and samples_seen % (int(est.validation_interval) * m) == 0:
                began_validation = time.perf_counter()
                try:
                    Xv, yv = validation
                    prediction = _scores(kind, Xv, coefficients, intercept, X,
                                         getattr(est, "gamma_", None),
                                         getattr(est, "prediction_batch_size", 256), deadline,
                                         kernel_counter=kernel_counter,
                                         kernel_spec=getattr(est, "_kernel_spec_", None))
                    value = float(np.mean(dwd_loss(yv * prediction)))
                    _check_deadline(deadline)
                    if not np.isfinite(value):
                        raise FloatingPointError("Nonfinite validation loss")
                    if value < best_loss:
                        best_loss = value
                        best = _snapshot(coefficients, intercept, updates, rng, samples_seen)
                    if reference is None or reference - value > est.tol * max(1.0, reference):
                        reference, misses = value, 0
                    else:
                        misses += 1
                    entry = {"iteration": updates, "epoch": samples_seen / m,
                             "n_samples_seen": samples_seen,
                             "validation_loss": value,
                             "elapsed_seconds": time.perf_counter() - began,
                             "selected": False, "patience_reference_loss": reference,
                             "patience_misses": misses}
                    history.append(entry)
                    if callback is not None:
                        callback(_snapshot(coefficients, intercept, updates, rng, samples_seen), copy.deepcopy(entry))
                finally:
                    validation_time += time.perf_counter() - began_validation
                _check_deadline(deadline)
                if (est.stopping_ == "validation" and samples_seen >= est.min_epochs * m
                        and misses >= est.patience):
                    stop = "validation_patience"
                    break
        except _Deadline:
            stop = "time_cap" if best is not None else "time_cap_no_validation"
            break
        except (FloatingPointError, OverflowError) as exc:
            stop, error = "numerical_failure", str(exc)
            break
    final = _snapshot(coefficients, intercept, updates, rng, samples_seen)
    selected = copy.deepcopy(best if best is not None else final)
    for entry in history:
        entry["selected"] = entry["iteration"] == selected["iteration"]
    return {"initial_state_": initial, "final_state_": final, "selected_state_": selected,
            "selected_iteration_": selected["iteration"], "n_updates_": updates,
            "selected_n_samples_seen_": selected["n_samples_seen"],
            "n_samples_seen_": samples_seen, "n_epochs_": samples_seen / m,
            "batch_size_": batch_size, "n_updates_per_epoch_": steps_per_epoch,
            "sampling_": "full" if full_batch else est.sampling,
            "learning_rate_clock_": "processed_observations",
            "learning_rate_B_": B, "history_": history,
            "stop_reason_": stop,
            "model_status_": "numerical_failure" if error else ("trained" if updates else "untrained_fallback"),
            "validation_available_": best is not None,
            "best_validation_loss_": None if best is None else best_loss,
            "update_seconds_": updates_time, "validation_checkpoint_seconds_": validation_time,
            "optimization_seconds_": time.perf_counter() - began,
            "kernel_evaluation_count_": kernel_counter[0], "error_": error,
            "rng_state_": copy.deepcopy(rng.bit_generator.state)}


class _BaseDWD(ClassifierMixin, BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def __sklearn_is_fitted__(self):
        return getattr(self, "_fitted_", False)

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]

    def _fit(self, X, y, validation_data, kind, callback, deadline=None):
        started = time.perf_counter()
        _clear(self)
        _controls(self)
        if kind == "kernel":
            _penalty(self)
            _integer(self.prediction_batch_size, "prediction_batch_size")
            _cache_controls(self)
        if callback is not None and not callable(callback):
            raise ValueError("checkpoint_callback must be callable or None")
        X, original_y, encoded, validation = _training_data(self, X, y, validation_data)
        _resolve_stopping(self, validation)
        if self.max_seconds is not None:
            local_deadline = started + self.max_seconds
            deadline = local_deadline if deadline is None else min(deadline, local_deadline)
        try:
            if kind == "kernel":
                prepared_at = time.perf_counter()
                self.kernel_preparation_evaluation_count_ = 0
                try:
                    spec = _set_kernel(self, X)
                    guard = lambda: _check_deadline(deadline)
                    preparation = spec.prepare(X, self.prediction_batch_size, guard)
                    self.max_kernel_diagonal_ = preparation["max_diagonal"]
                    self.kernel_preparation_evaluation_count_ = preparation["evaluation_count"]
                    self.kernel_validation_ = preparation["validation"]
                except _Deadline as exc:
                    self.kernel_preparation_evaluation_count_ = getattr(exc, "evaluation_count", 0)
                    if spec.kernel == "rbf":
                        # Its diagonal is identically one by definition; even
                        # an immediate cap can safely return the old diagnostic
                        # zero state without completing a diagonal traversal.
                        self.max_kernel_diagonal_ = 1.0
                        self.kernel_validation_ = "analytic_psd_family"
                    else:
                        self.model_status_, self.stop_reason_ = "kernel_preparation_timeout", "time_cap"
                        self.fit_seconds_ = time.perf_counter() - started
                        _fit_diagnostics(self)
                        raise TrainingError("Deadline reached before kernel validation completed",
                                            "kernel_preparation_timeout") from exc
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    self.kernel_preparation_evaluation_count_ = getattr(exc, "evaluation_count", 0)
                    raise
                finally:
                    self.kernel_preparation_seconds_ = time.perf_counter() - prepared_at
            result = _optimize(kind, X, encoded, validation, _generator(self.random_state),
                               self, deadline, callback)
        except (FloatingPointError, OverflowError, NumericalFailure) as exc:
            self.model_status_, self.stop_reason_ = "numerical_failure", "numerical_failure"
            self.error_ = str(exc)
            self.fit_seconds_ = time.perf_counter() - started
            _fit_diagnostics(self)
            raise NumericalFailure(str(exc)) from exc
        for key, value in result.items():
            setattr(self, key, value)
        if kind == "kernel":
            self.kernel_total_evaluation_count_ = (self.kernel_evaluation_count_ +
                                                  self.kernel_preparation_evaluation_count_)
        self.n_samples_fit_ = len(y)
        self.objective_normalization_ = "mean_over_actual_fitting_set"
        if self.model_status_ == "numerical_failure":
            self.fit_seconds_ = time.perf_counter() - started
            _fit_diagnostics(self)
            raise NumericalFailure(self.error_)
        state = self.selected_state_
        self.intercept_ = np.array([state["intercept"]])
        if kind == "linear":
            self.coef_ = state["coefficients"].reshape(1, -1).copy()
        else:
            self.dual_coef_ = state["coefficients"].reshape(1, -1).copy()
            self.basis_ = X.copy()
        self._fitted_ = True
        finished = time.perf_counter()
        if deadline is not None and finished >= deadline:
            self.stop_reason_ = "time_cap" if self.validation_available_ else "time_cap_no_validation"
        self.fit_seconds_ = finished - started
        _fit_diagnostics(self)
        return self


class LinearSGDDWD(_BaseDWD):
    """Unit-ball projected linear DWD with a separate unrestricted intercept.

    Each epoch draws n_samples observations independently with replacement.
    batch_size=1 gives single-observation steps, a larger positive integer
    gives averaged minibatches, and 'full' gives one full-gradient step.
    sampling='shuffle' optionally visits each row once in random order per epoch.
    With validation_data, retain the lowest post-update validation loss; otherwise
    retain the final iterate. Data are not automatically rescaled.

    stopping='auto' uses validation patience when validation_data is supplied,
    and the fixed epoch budget otherwise. 'fixed' disables patience while still
    monitoring supplied validation data and retaining its best checkpoint.
    'validation' requires explicit validation_data. n_iter_ counts SGD updates;
    n_epochs_completed_ counts processed observations divided by fitting size.
    criterion_reached_ reports the named termination_criterion_, never
    optimality. converged_=False is accompanied by optimality_checked_=False.

    optimizer_backend='python' is the default. 'auto' uses the optional native
    extension only for batch_size=1, contiguous aligned features, and no
    validation, deadline or callback; otherwise it records a Python fallback.
    'native' requires that supported configuration and an installed extension.
    optimizer_backend_ and optimizer_backend_reason_ describe the actual path.
    """
    def __init__(self, *, batch_size=1, sampling="with_replacement", max_epochs=200,
                 validation_interval=5, patience=3,
                 min_epochs=10, tol=1e-4, stopping="auto", random_state=None, max_seconds=None,
                 optimizer_backend="python"):
        self.batch_size = batch_size
        self.sampling = sampling
        self.max_epochs = max_epochs
        self.validation_interval = validation_interval
        self.patience = patience
        self.min_epochs = min_epochs
        self.tol = tol
        self.stopping = stopping
        self.random_state = random_state
        self.max_seconds = max_seconds
        self.optimizer_backend = optimizer_backend

    def fit(self, X, y, *, validation_data=None, checkpoint_callback=None):
        """Fit from zero; optional callback receives independent state/metric copies."""
        return self._fit(X, y, validation_data, "linear", checkpoint_callback)

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64, accept_sparse=False)
        return _scores("linear", X, self.coef_[0], self.intercept_[0])

    def objective(self, X, y):
        """Return the mean normalized DWD loss for supplied labeled observations."""
        score = self.decision_function(X)
        encoded = _evaluation_labels(self, y, len(score))
        return float(np.mean(dwd_loss(encoded * score)))


class KernelSGDDWD(_BaseDWD):
    """RKHS SGD for mean DWD loss plus lambd times the squared RKHS norm.

    Kernel values are evaluated on demand. No persistent Gram matrix is kept.
    ``lambd`` uses exactly the regularization convention of the dwd package.
    The default is 1e-6; the RKHS regularization gradient is 2*lambd*h.
    Named PSD kernels use analytic diagonals; custom/untrusted kernels require
    a transient training Gram check before fitting. Callable kernels accept
    two entire feature matrices and return the corresponding kernel matrix.
    Labels are incorporated into coefficient updates exactly once.
    batch_size controls averaged optimizer updates; prediction_batch_size
    bounds kernel scoring blocks without changing the update definition.
    stopping and fitting diagnostics follow LinearSGDDWD: epoch-limited finite
    fits are usable without an optimality certificate. n_iter_ counts SGD
    updates, not MM iterations; n_epochs_completed_ records sample-equivalent
    epochs. Explicit validation data also select the best checkpoint when
    stopping='fixed', which disables only patience-based early stopping.

    kernel_cache_bytes=0 disables training-column caching. A positive integer
    enables fit-local lazy LRU reuse for batch_size=1 and analytic built-in
    kernels only. The byte limit covers retained float64 column array payload,
    not Python metadata or kernel working memory. No Gram matrix is computed
    up front, and cached arrays are released when fitting exits. Fixed-budget
    updates are unchanged; wall-time-limited fits may reach a different update.
    optimizer_backend follows LinearSGDDWD. The native path additionally
    requires an analytic built-in kernel; automatic fallback is reported.
    """
    def __init__(self, *, lambd=1e-6, kernel="rbf", gamma="scale",
                 kernel_kws=None, batch_size=1,
                 sampling="with_replacement", max_epochs=200,
                 validation_interval=5, patience=3, min_epochs=10, tol=1e-4,
                 stopping="auto", random_state=None, max_seconds=None, prediction_batch_size=256,
                 kernel_cache_bytes=0, optimizer_backend="python"):
        self.lambd = lambd
        self.kernel = kernel
        self.gamma = gamma
        self.kernel_kws = kernel_kws
        self.batch_size = batch_size
        self.sampling = sampling
        self.max_epochs = max_epochs
        self.validation_interval = validation_interval
        self.patience = patience
        self.min_epochs = min_epochs
        self.tol = tol
        self.stopping = stopping
        self.random_state = random_state
        self.max_seconds = max_seconds
        self.prediction_batch_size = prediction_batch_size
        self.kernel_cache_bytes = kernel_cache_bytes
        self.optimizer_backend = optimizer_backend

    def fit(self, X, y, *, validation_data=None, checkpoint_callback=None):
        return self._fit(X, y, validation_data, "kernel", checkpoint_callback)

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64, accept_sparse=False)
        if self._kernel_spec_.kernel == "rbf" and self.gamma_ > 0:
            return rbf_prediction_scores(X, self.basis_, self.dual_coef_[0],
                                         self.intercept_[0], self.gamma_,
                                         self.prediction_batch_size)
        return _scores("kernel", X, self.dual_coef_[0], self.intercept_[0],
                       self.basis_, self.gamma_, self.prediction_batch_size,
                       kernel_spec=self._kernel_spec_)

    def rkhs_norm_squared(self):
        """Compute a^T K a in blocks; this optional diagnostic costs O(m^2 d)."""
        check_is_fitted(self)
        Ka = _scores("kernel", self.basis_, self.dual_coef_[0], 0.0,
                     self.basis_, self.gamma_, self.prediction_batch_size,
                     kernel_spec=self._kernel_spec_)
        value = float(self.dual_coef_[0] @ Ka)
        if not np.isfinite(value):
            raise NumericalFailure("Nonfinite RKHS norm diagnostic")
        # A tiny negative value can arise from accumulation roundoff for PSD K.
        tolerance = 64 * np.finfo(float).eps * max(1.0, float(np.sum(np.abs(self.dual_coef_[0] * Ka))))
        if value < -tolerance:
            raise NumericalFailure("Negative RKHS squared norm beyond roundoff")
        return max(0.0, value)

    def objective(self, X, y):
        """Evaluate mean loss plus the fitted function's RKHS penalty."""
        score = self.decision_function(X)
        encoded = _evaluation_labels(self, y, len(score))
        return float(np.mean(dwd_loss(encoded * score)) +
                     self.lambd_ * self.rkhs_norm_squared())


def _evaluation_labels(est, y, n):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n or not np.isin(y, est.classes_).all():
        raise ValueError("Labels must match observations and fitted classes")
    return np.where(y == est.classes_[1], 1.0, -1.0)


class TwoStageDWDClassifier(_BaseDWD):
    """Linear projected SGD, global absolute-score selection, then kernel SGD.

    Selection is label independent and does not impose class/side quotas.
    A one-class retained set raises SelectionError and preserves linear-stage
    diagnostics. Its kernel objective averages over the retained observations.
    batch_size and sampling apply to both stages with their actual fitting sizes.
    stopping applies to both stages as in LinearSGDDWD. stage_diagnostics_
    retains each stage's stopping and usability facts; n_iter_,
    returned_iteration_, n_epochs_completed_ and optimality flags are arrays in
    linear, kernel order. Aggregate cv_eligible_ requires both stages to be
    usable without wall-time truncation. A reached criterion is not optimality.
    kernel_cache_bytes optionally bounds lazy kernel-stage column storage as
    in KernelSGDDWD; the linear stage does not use this cache.
    optimizer_backend applies to both stages. stage_diagnostics_ reports each
    actual backend; optimizer_backend_ is 'mixed' if stage paths differ.
    """
    def __init__(self, *, retention_fraction=0.3, lambd=1e-6,
                 kernel="rbf", gamma="scale", kernel_kws=None,
                 batch_size=1, sampling="with_replacement",
                 max_epochs=200, validation_interval=5, patience=3, min_epochs=10,
                 tol=1e-4, stopping="auto", random_state=None, max_seconds=None,
                 linear_max_seconds=None, prediction_batch_size=256, kernel_cache_bytes=0,
                 optimizer_backend="python"):
        self.retention_fraction = retention_fraction
        self.lambd = lambd
        self.kernel = kernel
        self.gamma = gamma
        self.kernel_kws = kernel_kws
        self.batch_size = batch_size
        self.sampling = sampling
        self.max_epochs = max_epochs
        self.validation_interval = validation_interval
        self.patience = patience
        self.min_epochs = min_epochs
        self.tol = tol
        self.stopping = stopping
        self.random_state = random_state
        self.max_seconds = max_seconds
        self.linear_max_seconds = linear_max_seconds
        self.prediction_batch_size = prediction_batch_size
        self.kernel_cache_bytes = kernel_cache_bytes
        self.optimizer_backend = optimizer_backend

    def fit(self, X, y, *, validation_data=None, observation_ids=None,
            checkpoint_callback=None):
        started = time.perf_counter()
        _clear(self)
        _controls(self)
        _positive(self.retention_fraction, "retention_fraction")
        if self.retention_fraction > 1:
            raise ValueError("retention_fraction must be <= 1")
        _penalty(self)
        _integer(self.prediction_batch_size, "prediction_batch_size")
        _cache_controls(self)
        if self.linear_max_seconds is not None:
            _positive(self.linear_max_seconds, "linear_max_seconds")
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise ValueError("checkpoint_callback must be callable or None")
        X, y, encoded, validation = _training_data(self, X, y, validation_data)
        _resolve_stopping(self, validation)
        # Resolve data-dependent gamma on the complete training fold, before
        # screening; the retained kernel stage receives this fixed bandwidth.
        _set_kernel(self, X)
        ids = np.arange(len(y)) if observation_ids is None else np.asarray(observation_ids)
        if ids.ndim != 1 or len(ids) != len(y) or len(np.unique(ids)) != len(y):
            raise ValueError("observation_ids must be a unique one-dimensional ID per row")
        rng = _generator(self.random_state)
        self.stage_seeds_ = dict(zip(("linear_sgd", "selection_ties", "kernel_sgd"),
                                    (int(v) for v in rng.integers(0, 2**32, size=3, dtype=np.uint64))))
        deadline = None if self.max_seconds is None else started + self.max_seconds
        common = dict(batch_size=self.batch_size, sampling=self.sampling,
                      max_epochs=self.max_epochs,
                      validation_interval=self.validation_interval,
                      patience=self.patience, min_epochs=self.min_epochs, tol=self.tol,
                      stopping=self.stopping, optimizer_backend=self.optimizer_backend)
        self.linear_estimator_ = LinearSGDDWD(**common,
            random_state=self.stage_seeds_["linear_sgd"], max_seconds=self.linear_max_seconds)
        def callback_for(stage):
            if checkpoint_callback is None:
                return None
            return lambda state, metrics: checkpoint_callback(stage, state, metrics)
        try:
            self.linear_estimator_._fit(X, y, validation_data, "linear",
                                       callback_for("linear"), deadline)
        except NumericalFailure as exc:
            self.model_status_, self.stop_reason_ = "linear_stage_failure", "numerical_failure"
            self.fit_seconds_ = time.perf_counter() - started
            _two_stage_diagnostics(self)
            raise TrainingError(str(exc), "linear_stage_failure") from exc
        if self.linear_estimator_.model_status_ != "trained":
            self.model_status_, self.stop_reason_ = "linear_stage_failure", self.linear_estimator_.stop_reason_
            self.fit_seconds_ = time.perf_counter() - started
            _two_stage_diagnostics(self)
            raise TrainingError("The linear stage produced no trained finite state", "linear_stage_failure")
        began_selection = time.perf_counter()
        try:
            raw_scores = _scores("linear", X, self.linear_estimator_.coef_[0],
                                 self.linear_estimator_.intercept_[0],
                                 batch_size=self.prediction_batch_size, deadline=deadline)
            self.selection_scores_ = np.abs(raw_scores)
            self.selected_indices_ = select_near_boundary(raw_scores, self.retention_fraction,
                _generator(self.stage_seeds_["selection_ties"]), observation_ids=ids)
            self.selected_ids_ = ids[self.selected_indices_].copy()
            selected_y = y[self.selected_indices_]
            self.selected_class_counts_ = np.array([np.sum(selected_y == c) for c in self.classes_])
            self.selection_seconds_ = time.perf_counter() - began_selection
            self.n_retained_ = len(self.selected_indices_)
            self.n_samples_fit_ = len(y)
            if np.any(self.selected_class_counts_ == 0):
                self.model_status_, self.stop_reason_ = "single_class_selection", "selection_failure"
                self.fit_seconds_ = time.perf_counter() - started
                _two_stage_diagnostics(self)
                raise SelectionError("The globally selected subset contains only one class")
            _check_deadline(deadline)
        except _Deadline as exc:
            self.model_status_, self.stop_reason_ = "kernel_stage_unrun", "time_cap"
            self.fit_seconds_ = time.perf_counter() - started
            _two_stage_diagnostics(self)
            raise TrainingError("Pipeline deadline reached before kernel fitting", "kernel_stage_unrun") from exc
        except (FloatingPointError, OverflowError) as exc:
            self.model_status_, self.stop_reason_ = "numerical_failure", "numerical_failure"
            self.fit_seconds_ = time.perf_counter() - started
            _two_stage_diagnostics(self)
            raise NumericalFailure(str(exc)) from exc
        finally:
            self.selection_seconds_ = time.perf_counter() - began_selection
        kernel = KernelSGDDWD(**common, lambd=self.lambd_,
            kernel=self.kernel_, gamma=self.gamma_ if self.gamma_ is not None else self.gamma,
            kernel_kws=copy.deepcopy(self._kernel_input_kws_),
            random_state=self.stage_seeds_["kernel_sgd"], prediction_batch_size=self.prediction_batch_size,
            kernel_cache_bytes=self.kernel_cache_bytes)
        # Keep the kernel object for diagnostics after failures; fittedness is
        # controlled explicitly, never inferred from the existence of this object.
        self.kernel_estimator_ = kernel
        try:
            kernel._fit(X[self.selected_indices_], selected_y, validation_data, "kernel",
                        callback_for("kernel"), deadline)
        except NumericalFailure:
            self.model_status_, self.stop_reason_ = "numerical_failure", "numerical_failure"
            self.fit_seconds_ = time.perf_counter() - started
            _two_stage_diagnostics(self)
            raise
        except TrainingError as exc:
            self.model_status_, self.stop_reason_ = exc.status, kernel.stop_reason_
            self.fit_seconds_ = time.perf_counter() - started
            _two_stage_diagnostics(self)
            raise
        self.model_status_, self.stop_reason_ = kernel.model_status_, kernel.stop_reason_
        self.fit_seconds_ = time.perf_counter() - started
        self.objective_normalization_ = "mean_over_retained_set"
        self._fitted_ = True
        _two_stage_diagnostics(self)
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64, accept_sparse=False)
        return self.kernel_estimator_.decision_function(X)
