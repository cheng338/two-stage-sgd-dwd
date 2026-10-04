"""Fold-local fixed-epoch evaluation for the MNIST tuning example.

The caller supplies already scaled pixels and saved CV indices. This helper
never loads test data, learns a scaler, or uses validation labels for fitting.
Linear screening is shared only across candidates with identical training
indices, epoch budget, batch size, seed, and optimizer identity. Kernel models are discarded after
scoring; only one current linear model and its training scores per fold remain.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import numbers
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_array

from two_stage_sgd_dwd import KernelSGDDWD, LinearSGDDWD, SelectionError, TrainingError
from two_stage_sgd_dwd import __version__ as _PACKAGE_VERSION
from two_stage_sgd_dwd._core import dwd_loss, select_near_boundary
from two_stage_sgd_dwd._estimators import _generator
from two_stage_sgd_dwd._kernels import resolve_kernel


def _integer(value, name, minimum=1):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, numbers.Integral) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, numbers.Real) or
            not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _ids_hash(ids):
    """SHA256 of the ordered ID list, encoded as compact UTF-8 JSON."""
    payload = json.dumps(np.asarray(ids).tolist(), ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def execution_identity(optimizer_backend="python"):
    """Return portable provenance for the requested optimizer implementation.

    Python execution does not import or depend on an unused native extension.
    Auto/native identities bind the actually importable extension's bytes; an
    unavailable extension is an explicit auto fallback or a native-only error.
    The caller can persist this dictionary to reject incompatible resumes.
    """
    if not isinstance(optimizer_backend, str) or optimizer_backend not in ("python", "auto", "native"):
        raise ValueError("optimizer_backend must be 'python', 'auto', or 'native'")
    result = {"requested_backend": optimizer_backend, "extension_available": None,
              "extension_filename": None, "extension_sha256": None,
              "package_version": _PACKAGE_VERSION}
    if optimizer_backend == "python":
        return result
    try:
        native = importlib.import_module("two_stage_sgd_dwd._sgd_accel")
    except (ImportError, OSError) as exc:
        if optimizer_backend == "native":
            raise ImportError("optimizer_backend='native' requires the optional compiled _sgd_accel extension") from exc
        result["extension_available"] = False
        return result
    path = Path(native.__file__)
    result.update(extension_available=True, extension_filename=path.name,
                  extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


class FoldEvaluator:
    """Evaluate RBF candidates with independently fitted CV screening stages.

    Parameters ``X``, ``y``, and ``ids`` are copied. Each fold must partition all
    supplied rows into disjoint training and validation indices, and validation
    folds must cover every row exactly once. Labels must describe two classes.
    IDs must be unique, mutually comparable, JSON-serializable scalar values.
    Pass pixels divided by 255; no transformation is performed here.

    ``evaluate`` accepts scalar ``lambd``, ``gamma`` (nonnegative numeric or
    ``'scale'``), and ``retention_fraction``; optional ``kernel`` must be ``'rbf'``.
    Returned records contain JSON-native dictionaries/lists/scalars. The top
    level has ``params``, ``eligible`` (bool), ``mean_score`` (float or None),
    and ``folds``. Each fold records float accuracy/loss (None after selection
    failure), integer counts/updates/seeds, resolved float gamma, ordered-ID
    SHA256, eligibility/status, and component seconds. ``linear_seconds_reused``
    is the original cached linear fit plus training-score time, even on a cache
    hit; ``linear_seconds_charged`` is zero on a hit. These component times are
    not a fresh complete-pipeline wall-time measurement.

    Sampling is always with replacement. No validation_data or deadlines are
    passed to either optimizer, so successful fits must finish fixed epochs.
    Optimizer defaults remain ``optimizer_backend='python'`` and
    ``kernel_cache_bytes=0``. The backend applies to both stages; the optional
    lazy byte-bounded training-column cache applies only to the kernel stage.
    Positive cache capacity and explicit native execution require batch_size=1.
    Auto reports its actual native path or Python fallback in each stage record.
    Changing max_epochs, random_state, batch_size, or requested backend invalidates
    each affected fold's linear cache on its next use; changing kernel cache
    capacity does not. Native implementation identity must remain unchanged
    throughout an evaluator's lifetime. Numerical/unexpected errors propagate;
    only one-class retained selections become ordinary ineligible records.
    """

    def __init__(self, X, y, ids, folds, *, max_epochs, random_state=42, batch_size=1,
                 optimizer_backend="python", kernel_cache_bytes=0):
        self._X = check_array(X, dtype=np.float64, accept_sparse=False,
                              ensure_min_samples=2, copy=True)
        self._y = np.asarray(y).copy()
        if self._y.ndim != 1 or len(self._y) != len(self._X):
            raise ValueError("y must contain one label per input row")
        check_classification_targets(self._y)
        self._classes = unique_labels(self._y)
        if len(self._classes) != 2:
            raise ValueError("FoldEvaluator requires exactly two classes")
        self._ids = np.asarray(ids).copy()
        if self._ids.ndim != 1 or len(self._ids) != len(self._X):
            raise ValueError("ids must contain one unique scalar ID per input row")
        try:
            if len(np.unique(self._ids)) != len(self._ids):
                raise ValueError("ids must be unique")
            _ids_hash(self._ids)
        except (TypeError, ValueError) as exc:
            raise ValueError("ids must be unique, comparable, finite JSON scalar values") from exc
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.optimizer_backend = optimizer_backend
        self.kernel_cache_bytes = kernel_cache_bytes
        self._execution_identities = {}
        self.execution_identity_ = None
        self._controls()
        self._folds = self._validate_folds(folds)
        for array in (self._X, self._y, self._ids):
            array.setflags(write=False)
        self._linear_cache = {}
        self.linear_fit_count_ = 0

    def _controls(self):
        self.max_epochs = _integer(self.max_epochs, "max_epochs")
        self.random_state = _integer(self.random_state, "random_state", 0)
        if not (isinstance(self.batch_size, str) and self.batch_size == "full"):
            self.batch_size = _integer(self.batch_size, "batch_size")
        self.kernel_cache_bytes = _integer(self.kernel_cache_bytes, "kernel_cache_bytes", 0)
        if self.kernel_cache_bytes and self.batch_size != 1:
            raise ValueError("Positive kernel_cache_bytes requires batch_size=1")
        if self.optimizer_backend == "native" and self.batch_size != 1:
            raise ValueError("optimizer_backend='native' requires batch_size=1")
        self._check_execution_identity()
        return {"max_epochs": self.max_epochs, "batch_size": self.batch_size,
                "sampling": "with_replacement", "stopping": "fixed",
                "optimizer_backend": self.optimizer_backend}

    def _check_execution_identity(self):
        current = execution_identity(self.optimizer_backend)
        previous = self._execution_identities.get(self.optimizer_backend)
        if previous is not None and current != previous:
            raise RuntimeError("Optimizer execution identity changed during fold evaluation")
        self._execution_identities[self.optimizer_backend] = deepcopy(current)
        self.execution_identity_ = deepcopy(current)
        return current

    def _validate_folds(self, folds):
        validated = []
        validation_counts = np.zeros(len(self._X), dtype=np.intp)
        for pair in folds:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each fold must contain training and validation indices")
            converted = []
            for indices in pair:
                array = np.asarray(indices)
                if (array.ndim != 1 or array.dtype.kind not in "iu" or
                        len(array) == 0 or np.any(array < 0) or
                        np.any(array >= len(self._X)) or
                        len(np.unique(array)) != len(array)):
                    raise ValueError("Fold indices must be nonempty unique in-range integer vectors")
                converted.append(array.astype(np.intp, copy=True))
            train, validation = converted
            if (np.intersect1d(train, validation).size or
                    len(train) + len(validation) != len(self._X)):
                raise ValueError("Each fold must be a disjoint partition of all supplied rows")
            if len(unique_labels(self._y[train])) != 2:
                raise ValueError("Each training fold must contain both classes")
            validation_counts[validation] += 1
            train.setflags(write=False)
            validation.setflags(write=False)
            validated.append((train, validation))
        if len(validated) < 2 or not np.all(validation_counts == 1):
            raise ValueError("Validation folds must cover every supplied row exactly once")
        return validated

    def cv_fold_identity(self):
        """Return independent JSON index pairs in the original fold order."""
        return [[train.tolist(), validation.tolist()] for train, validation in self._folds]

    def _check_complete(self, model, sample_count):
        actual_batch = sample_count if self.batch_size == "full" else min(self.batch_size, sample_count)
        expected_updates = ((sample_count + actual_batch - 1) // actual_batch) * self.max_epochs
        if (not model.cv_eligible_ or model.model_status_ != "trained" or model.stop_reason_ != "epoch_cap" or
                model.n_updates_ != expected_updates or
                model.n_samples_seen_ != self.max_epochs * sample_count):
            raise TrainingError("CV fitting did not complete its fixed epoch budget",
                                status="incomplete_cv_fit")

    @staticmethod
    def _fit_diagnostics(model):
        """Keep usable budget-limited fits separate from optimality claims."""
        result = {"model_status": model.model_status_, "stop_reason": model.stop_reason_,
                "termination_criterion": model.termination_criterion_,
                "criterion_reached": bool(model.criterion_reached_),
                "cap_reached": bool(model.cap_reached_), "fit_usable": bool(model.fit_usable_),
                "cv_eligible": bool(model.cv_eligible_),
                "optimality_checked": bool(model.optimality_checked_),
                "optimality_met": bool(model.optimality_met_),
                "converged": bool(model.converged_),
                "n_epochs_completed": float(model.n_epochs_completed_),
                "n_updates": int(model.n_updates_),
                "n_samples_seen": int(model.n_samples_seen_),
                "optimizer_backend": model.optimizer_backend_,
                "optimizer_backend_reason": model.optimizer_backend_reason_}
        if hasattr(model, "kernel_cache_bytes"):
            result.update(kernel_cache_bytes=int(model.kernel_cache_bytes),
                          kernel_evaluation_count=int(model.kernel_evaluation_count_),
                          kernel_cache_hits=int(model.kernel_cache_hits_),
                          kernel_cache_misses=int(model.kernel_cache_misses_),
                          kernel_cache_evictions=int(model.kernel_cache_evictions_),
                          kernel_cache_peak_bytes=int(model.kernel_cache_peak_bytes_),
                          kernel_cache_capacity_columns=int(model.kernel_cache_capacity_columns_))
        return result

    def _linear_stage(self, fold_index, controls):
        cache_key = (self.max_epochs, self.random_state, self.batch_size,
                     json.dumps(self.execution_identity_, sort_keys=True, separators=(",", ":")))
        cached = self._linear_cache.get(fold_index)
        if cached is not None and cached["key"] == cache_key:
            return cached, True
        train, _ = self._folds[fold_index]
        seed = int(np.random.SeedSequence([self.random_state, fold_index]).generate_state(1)[0])
        stage_seeds = dict(zip(("linear_sgd", "selection_ties", "kernel_sgd"),
            (int(value) for value in _generator(seed).integers(0, 2**32, size=3, dtype=np.uint64))))
        began = perf_counter()
        model = LinearSGDDWD(**controls, random_state=stage_seeds["linear_sgd"])
        model.fit(self._X[train], self._y[train])
        self._check_complete(model, len(train))
        raw_scores = model.decision_function(self._X[train])
        cached = {"key": cache_key, "model": model, "raw_scores": raw_scores,
                  "seconds": float(perf_counter() - began), "seed": seed,
                  "stage_seeds": stage_seeds}
        self._linear_cache[fold_index] = cached
        self.linear_fit_count_ += 1
        return cached, False

    @staticmethod
    def _parameters(params):
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary of scalar candidate values")
        required = {"lambd", "gamma", "retention_fraction"}
        if not required <= params.keys() or set(params) - required - {"kernel"}:
            raise ValueError("params must provide lambd, gamma, retention_fraction, and optionally kernel")
        if params.get("kernel", "rbf") != "rbf":
            raise ValueError("FoldEvaluator supports only kernel='rbf'")
        fraction = _positive(params["retention_fraction"], "retention_fraction")
        if fraction > 1.0:
            raise ValueError("retention_fraction must be <= 1")
        gamma = params["gamma"]
        if isinstance(gamma, str):
            if gamma != "scale":
                raise ValueError("gamma must be nonnegative numeric or 'scale'")
        elif (isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, numbers.Real)
              or not np.isfinite(gamma) or gamma < 0):
            raise ValueError("gamma must be nonnegative numeric or 'scale'")
        else:
            gamma = float(gamma)
        return {"kernel": "rbf", "lambd": _positive(params["lambd"], "lambd"),
                "gamma": gamma, "retention_fraction": fraction}

    def evaluate(self, params):
        """Evaluate one candidate, preserving all fold selection failures."""
        params = self._parameters(params)
        controls = self._controls()
        records = []
        for fold_index, (train, validation) in enumerate(self._folds):
            # Resolve data-dependent width on the full training fold, as does
            # TwoStageDWDClassifier, never on the retained subset or validation.
            gamma = resolve_kernel(self._X[train], "rbf", params["gamma"]).gamma
            cached, cache_hit = self._linear_stage(fold_index, controls)
            stage_seeds = cached["stage_seeds"]
            record = {"fold": fold_index, "seed": cached["seed"],
                      "stage_seeds": deepcopy(stage_seeds), "gamma_resolved": float(gamma),
                      "n_train": len(train), "n_validation": len(validation),
                      "accuracy": None, "dwd_loss": None, "eligible": False,
                      "linear_updates": int(cached["model"].n_updates_),
                      "linear_fit": self._fit_diagnostics(cached["model"]),
                      "kernel_fit": None,
                      "kernel_updates": 0, "kernel_seconds": 0.0, "score_seconds": 0.0,
                      "linear_seconds_reused": cached["seconds"],
                      "linear_seconds_charged": 0.0 if cache_hit else cached["seconds"],
                      "linear_cache_hit": cache_hit}
            began_selection = perf_counter()
            try:
                selected = select_near_boundary(cached["raw_scores"], params["retention_fraction"],
                    _generator(stage_seeds["selection_ties"]), observation_ids=self._ids[train])
                selected_rows = train[selected]
                counts = [int(np.sum(self._y[selected_rows] == label)) for label in self._classes]
                record.update(n_retained=len(selected), selected_class_counts=counts,
                              selected_ids_sha256=_ids_hash(self._ids[selected_rows]))
                if min(counts) == 0:
                    raise SelectionError("The globally selected subset contains only one class")
            except SelectionError as exc:
                record.update(status=exc.status, error=str(exc))
                records.append(record)
                continue
            finally:
                record["selection_seconds"] = float(perf_counter() - began_selection)
            model = KernelSGDDWD(**controls, lambd=params["lambd"], kernel="rbf", gamma=gamma,
                                 random_state=stage_seeds["kernel_sgd"],
                                 kernel_cache_bytes=self.kernel_cache_bytes)
            model.fit(self._X[selected_rows], self._y[selected_rows])
            self._check_complete(model, len(selected_rows))
            record["kernel_seconds"] = float(model.fit_seconds_)
            record["kernel_updates"] = int(model.n_updates_)
            record["kernel_fit"] = self._fit_diagnostics(model)
            began_score = perf_counter()
            scores = model.decision_function(self._X[validation])
            predicted = model.classes_[(scores >= 0).astype(int)]
            encoded = np.where(self._y[validation] == model.classes_[1], 1.0, -1.0)
            with np.errstate(over="raise", invalid="raise"):
                loss = float(np.mean(dwd_loss(encoded * scores)))
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite cross-validation DWD loss")
            correct = int(np.sum(predicted == self._y[validation]))
            record.update(accuracy=float(correct / len(validation)), dwd_loss=loss,
                          n_correct=correct, eligible=True, status="trained",
                          kernel_stop_reason=model.stop_reason_,
                          score_seconds=float(perf_counter() - began_score))
            records.append(record)
        eligible = all(record["eligible"] for record in records)
        # Check once at candidate boundaries, never in an optimizer update. A
        # changed extension cannot produce a completed/cache-compatible record.
        self._check_execution_identity()
        return {"params": params, "eligible": eligible,
                "mean_score": float(np.mean([record["accuracy"] for record in records])) if eligible else None,
                "folds": records, "execution_identity": deepcopy(self.execution_identity_)}
