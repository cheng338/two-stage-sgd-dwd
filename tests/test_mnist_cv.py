"""Tiny deterministic checks of the fold-local MNIST evaluation helper."""

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.model_selection import StratifiedKFold

from two_stage_sgd_dwd import (
    KernelSGDDWD, LinearSGDDWD, NumericalFailure, TrainingError, TwoStageDWDClassifier,
)


_HELPER_PATH = Path(__file__).resolve().parents[1] / "examples" / "_mnist_cv.py"
_SPEC = importlib.util.spec_from_file_location("mnist_cv_example", _HELPER_PATH)
mnist_cv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mnist_cv)


def small_data():
    magnitude = np.linspace(0.1, 0.9, 16)
    positive = np.column_stack((magnitude, np.zeros(16)))
    X = np.vstack((-positive, positive))
    y = np.array([7] * 16 + [8] * 16)
    ids = np.array([f"train:{index}" for index in range(len(y))])
    folds = list(StratifiedKFold(2, shuffle=True, random_state=17).split(X, y))
    return X, y, ids, folds


@pytest.mark.parametrize("gamma", [0.3, "scale"])
@pytest.mark.parametrize("lambd", [1e-6, 0.02])
def test_cached_evaluation_matches_public_pipeline_bitwise_for_retention_and_width(
        gamma, lambd, monkeypatch):
    X, y, ids, folds = small_data()
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=4, random_state=13)
    captured = []

    def capture_kernel(**kwargs):
        model = KernelSGDDWD(**kwargs)
        captured.append(model)
        return model

    monkeypatch.setattr(mnist_cv, "KernelSGDDWD", capture_kernel)
    for fraction in (0.75, 1.0):
        params = {"lambd": lambd, "gamma": gamma, "retention_fraction": fraction}
        captured.clear()
        record = evaluator.evaluate(params)
        assert record["eligible"]
        assert len(captured) == len(folds)
        for fold_index, (train, validation) in enumerate(folds):
            seed = int(np.random.SeedSequence([13, fold_index]).generate_state(1)[0])
            direct = TwoStageDWDClassifier(**params, max_epochs=4, random_state=seed,
                                           batch_size=1, sampling="with_replacement")
            direct.fit(X[train], y[train], observation_ids=ids[train])
            actual = captured[fold_index]
            expected = direct.kernel_estimator_
            assert_array_equal(actual.basis_, X[train][direct.selected_indices_])
            assert_array_equal(actual.dual_coef_, expected.dual_coef_)
            assert_array_equal(actual.intercept_, expected.intercept_)
            assert_array_equal(actual.decision_function(X[validation]), direct.decision_function(X[validation]))
            assert_array_equal(actual.predict(X[validation]), direct.predict(X[validation]))
            fold_record = record["folds"][fold_index]
            assert fold_record["seed"] == seed
            assert fold_record["stage_seeds"] == direct.stage_seeds_
            assert fold_record["gamma_resolved"] == direct.gamma_
            assert fold_record["selected_ids_sha256"] == mnist_cv._ids_hash(direct.selected_ids_)
            assert fold_record["accuracy"] == np.mean(direct.predict(X[validation]) == y[validation])
            assert fold_record["linear_updates"] == direct.linear_estimator_.n_updates_
            assert fold_record["kernel_updates"] == expected.n_updates_
            for stage in ("linear_fit", "kernel_fit"):
                diagnostics = fold_record[stage]
                assert diagnostics["cap_reached"] and diagnostics["criterion_reached"]
                assert diagnostics["fit_usable"] and diagnostics["cv_eligible"]
                assert diagnostics["termination_criterion"] == "fixed_epochs"
                assert diagnostics["n_epochs_completed"] == 4
                assert not diagnostics["optimality_checked"]
                assert not diagnostics["optimality_met"] and not diagnostics["converged"]
        assert record["mean_score"] == np.mean([fold["accuracy"] for fold in record["folds"]])
        json.dumps(record, allow_nan=False)
    assert evaluator.linear_fit_count_ == len(folds)
    assert all(fold["linear_cache_hit"] for fold in record["folds"])
    assert all(fold["linear_seconds_charged"] == 0 for fold in record["folds"])


def test_linear_fits_are_reused_across_all_kernel_candidates_and_refresh_with_controls(monkeypatch):
    X, y, ids, folds = small_data()
    calls = []
    original_fit = LinearSGDDWD.fit

    def observed_fit(model, features, labels, **kwargs):
        calls.append((features.copy(), labels.copy(), kwargs.copy()))
        return original_fit(model, features, labels, **kwargs)

    monkeypatch.setattr(LinearSGDDWD, "fit", observed_fit)
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2, random_state=9)
    for gamma in (0.2, "scale"):
        for lambd in (1e-6, 0.03):
            evaluator.evaluate({"lambd": lambd, "gamma": gamma, "retention_fraction": 1.0})
    assert len(calls) == len(folds)
    for (features, labels, kwargs), (train, _) in zip(calls, folds):
        assert_array_equal(features, X[train])
        assert_array_equal(labels, y[train])
        assert "validation_data" not in kwargs
    evaluator.max_epochs = 3
    evaluator.evaluate({"lambd": 1e-6, "gamma": 0.2, "retention_fraction": 1.0})
    assert len(calls) == 2 * len(folds)
    evaluator.random_state = 10
    evaluator.evaluate({"lambd": 1e-6, "gamma": 0.2, "retention_fraction": 1.0})
    assert len(calls) == 3 * len(folds)
    assert len(evaluator._linear_cache) == len(folds)


def test_selection_failure_is_recorded_but_numerical_failure_propagates(monkeypatch):
    X, y, ids, folds = small_data()
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2)
    record = evaluator.evaluate({"lambd": 1e-6, "gamma": 0.2, "retention_fraction": 0.01})
    assert not record["eligible"]
    assert record["mean_score"] is None
    assert len(record["folds"]) == len(folds)
    assert all(fold["status"] == "single_class_selection" for fold in record["folds"])
    assert all(fold["n_retained"] == 1 and fold["kernel_updates"] == 0 for fold in record["folds"])
    json.dumps(record, allow_nan=False)

    def failed_fit(*args, **kwargs):
        raise NumericalFailure("deliberate failure")

    monkeypatch.setattr(KernelSGDDWD, "fit", failed_fit)
    with pytest.raises(NumericalFailure, match="deliberate"):
        evaluator.evaluate({"lambd": 1e-6, "gamma": 0.2, "retention_fraction": 1.0})


def test_both_stages_fit_only_their_fold_training_rows_and_labels(monkeypatch):
    X, y, ids, folds = small_data()
    original_linear, original_kernel = LinearSGDDWD.fit, KernelSGDDWD.fit
    row_index = {tuple(row): index for index, row in enumerate(X)}
    calls = {"linear": [], "kernel": []}

    def observe(name, original):
        def fit(model, features, labels, **kwargs):
            rows = np.array([row_index[tuple(row)] for row in features])
            calls[name].append(rows)
            fold_index = len(calls[name]) - 1
            train, validation = folds[fold_index]
            assert np.isin(rows, train).all()
            assert not np.isin(rows, validation).any()
            assert_array_equal(labels, y[rows])
            assert "validation_data" not in kwargs
            return original(model, features, labels, **kwargs)
        return fit

    monkeypatch.setattr(LinearSGDDWD, "fit", observe("linear", original_linear))
    monkeypatch.setattr(KernelSGDDWD, "fit", observe("kernel", original_kernel))
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2)
    evaluator.evaluate({"lambd": 1e-6, "gamma": "scale", "retention_fraction": 0.75})
    assert len(calls["linear"]) == len(calls["kernel"]) == len(folds)


def test_incomplete_fixed_budget_cannot_be_ranked(monkeypatch):
    X, y, ids, folds = small_data()
    original = KernelSGDDWD.fit

    def incomplete(model, *args, **kwargs):
        original(model, *args, **kwargs)
        model.n_updates_ -= 1
        return model

    monkeypatch.setattr(KernelSGDDWD, "fit", incomplete)
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2)
    with pytest.raises(TrainingError, match="fixed epoch"):
        evaluator.evaluate({"lambd": 1e-6, "gamma": 0.2, "retention_fraction": 1.0})


def test_fold_inputs_are_copied_and_identity_is_independent():
    X, y, ids, folds = small_data()
    originals = X.copy(), y.copy(), ids.copy()
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2)
    identity = evaluator.cv_fold_identity()
    expected = [[train.tolist(), validation.tolist()] for train, validation in folds]
    assert identity == expected
    identity[0][0][0] = -100
    assert evaluator.cv_fold_identity() == expected
    X[:] = 0
    y[:] = 9
    ids[:] = "changed"
    for train, validation in folds:
        train[:] = 0
        validation[:] = 0
    record = evaluator.evaluate({"lambd": 1e-6, "gamma": 0.2, "retention_fraction": 1.0})
    assert record["eligible"]
    assert_array_equal(evaluator._X, originals[0])
    assert_array_equal(evaluator._y, originals[1])
    assert_array_equal(evaluator._ids, originals[2])


@pytest.mark.parametrize("invalid", ["overlap", "missing", "duplicate", "float", "out_of_range", "repeat_validation"])
def test_invalid_fold_boundaries_are_rejected(invalid):
    X, y, ids, folds = small_data()
    folds = [(train.copy(), validation.copy()) for train, validation in folds]
    train, validation = folds[0]
    if invalid == "overlap":
        train[0] = validation[0]
    elif invalid == "missing":
        folds[0] = (train[1:], validation)
    elif invalid == "duplicate":
        train[0] = train[1]
    elif invalid == "float":
        folds[0] = (train.astype(float), validation)
    elif invalid == "out_of_range":
        train[0] = len(X)
    else:
        folds[1] = folds[0]
    with pytest.raises(ValueError):
        mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2)


@pytest.mark.parametrize("which", ["features", "labels", "ids", "epochs", "seed", "batch"])
def test_nonfinite_or_invalid_constructor_inputs_are_rejected(which):
    X, y, ids, folds = small_data()
    controls = {"max_epochs": 2}
    if which == "features":
        X[0, 0] = np.inf
    elif which == "labels":
        y = y.astype(float)
        y[0] = np.nan
    elif which == "ids":
        ids = np.arange(len(y), dtype=float)
        ids[0] = np.nan
    elif which == "epochs":
        controls["max_epochs"] = 0
    elif which == "seed":
        controls["random_state"] = -1
    else:
        controls["batch_size"] = True
    with pytest.raises(ValueError):
        mnist_cv.FoldEvaluator(X, y, ids, folds, **controls)


@pytest.mark.parametrize("update", [
    {"lambd": np.nan}, {"lambd": 0}, {"gamma": -1}, {"gamma": "auto"},
    {"gamma": np.inf}, {"retention_fraction": 0}, {"retention_fraction": 1.1},
    {"kernel": "linear"}, {"unknown": 1},
])
def test_invalid_candidate_is_rejected_before_fitting(update):
    X, y, ids, folds = small_data()
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2)
    params = {"lambd": 1e-6, "gamma": 0.2, "retention_fraction": 1.0, **update}
    with pytest.raises(ValueError):
        evaluator.evaluate(params)
    assert evaluator.linear_fit_count_ == 0


def test_python_execution_identity_is_portable_and_never_imports_native(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Python execution must not inspect an unused extension")

    monkeypatch.setattr(mnist_cv.importlib, "import_module", unexpected)
    identity = mnist_cv.execution_identity("python")
    assert identity == {"requested_backend": "python", "extension_available": None,
                        "extension_filename": None, "extension_sha256": None,
                        "package_version": mnist_cv._PACKAGE_VERSION}
    json.dumps(identity, allow_nan=False)


def test_auto_identity_binds_the_actual_extension_bytes_or_portable_absence():
    identity = mnist_cv.execution_identity("auto")
    if not identity["extension_available"]:
        assert identity["extension_filename"] is None
        assert identity["extension_sha256"] is None
        with pytest.raises(ImportError, match="optional compiled"):
            mnist_cv.execution_identity("native")
        return
    module = mnist_cv.importlib.import_module("two_stage_sgd_dwd._sgd_accel")
    path = Path(module.__file__)
    assert identity["extension_filename"] == path.name
    assert identity["extension_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not Path(identity["extension_filename"]).is_absolute()
    native = mnist_cv.execution_identity("native")
    assert {k: v for k, v in native.items() if k != "requested_backend"} == {
        k: v for k, v in identity.items() if k != "requested_backend"}


@pytest.mark.parametrize("backend", ["python", "auto", "native"])
@pytest.mark.parametrize("cache_bytes", [0, 512])
@pytest.mark.parametrize("gamma", [0.0, "scale"])
def test_accelerated_folds_match_public_python_and_keep_stage_rng(backend, cache_bytes, gamma, monkeypatch):
    if backend == "native" and not mnist_cv.execution_identity("auto")["extension_available"]:
        pytest.skip("Optional native extension unavailable")
    X, y, ids, folds = small_data()
    linear_models, kernel_models = [], []

    def capture(model_class, destination):
        def factory(**kwargs):
            model = model_class(**kwargs)
            destination.append(model)
            return model
        return factory

    monkeypatch.setattr(mnist_cv, "LinearSGDDWD", capture(LinearSGDDWD, linear_models))
    monkeypatch.setattr(mnist_cv, "KernelSGDDWD", capture(KernelSGDDWD, kernel_models))
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=4, random_state=13,
                                      optimizer_backend=backend, kernel_cache_bytes=cache_bytes)
    params = {"lambd": 0.02, "gamma": gamma, "retention_fraction": 0.75}
    record = evaluator.evaluate(params)
    assert record["eligible"]
    assert record["execution_identity"] == evaluator.execution_identity_
    assert len(linear_models) == len(kernel_models) == len(folds)
    for fold_index, (train, validation) in enumerate(folds):
        seed = int(np.random.SeedSequence([13, fold_index]).generate_state(1)[0])
        reference = TwoStageDWDClassifier(**params, max_epochs=4, random_state=seed,
                                          optimizer_backend="python", kernel_cache_bytes=0).fit(
                                              X[train], y[train], observation_ids=ids[train])
        actual_linear, actual_kernel = linear_models[fold_index], kernel_models[fold_index]
        for actual, expected in ((actual_linear, reference.linear_estimator_),
                                 (actual_kernel, reference.kernel_estimator_)):
            assert_array_equal(actual.final_state_["coefficients"], expected.final_state_["coefficients"])
            assert actual.final_state_["intercept"] == expected.final_state_["intercept"]
            assert actual.rng_state_ == expected.rng_state_
            assert actual.n_updates_ == expected.n_updates_
            assert actual.optimizer_backend == backend
        assert not hasattr(actual_linear, "kernel_cache_bytes")
        assert_array_equal(actual_kernel.basis_, reference.kernel_estimator_.basis_)
        assert_array_equal(actual_kernel.decision_function(X[validation]), reference.decision_function(X[validation]))
        fold = record["folds"][fold_index]
        assert fold["selected_ids_sha256"] == mnist_cv._ids_hash(reference.selected_ids_)
        assert fold["stage_seeds"] == reference.stage_seeds_
        assert fold["linear_fit"]["optimizer_backend"] == actual_linear.optimizer_backend_
        assert fold["kernel_fit"]["optimizer_backend_reason"] == actual_kernel.optimizer_backend_reason_
        diagnostics = fold["kernel_fit"]
        assert diagnostics["kernel_cache_bytes"] == cache_bytes
        assert diagnostics["kernel_cache_peak_bytes"] <= cache_bytes
        assert diagnostics["kernel_evaluation_count"] == actual_kernel.kernel_evaluation_count_
        if cache_bytes:
            assert diagnostics["kernel_cache_hits"] > 0
            assert diagnostics["kernel_cache_evictions"] > 0
        else:
            assert diagnostics["kernel_cache_hits"] == diagnostics["kernel_cache_peak_bytes"] == 0
    second = evaluator.evaluate({**params, "lambd": 0.01})
    assert all(fold["linear_cache_hit"] for fold in second["folds"])
    assert len(linear_models) == len(folds)
    json.dumps(record, allow_nan=False)


def test_cache_capacity_does_not_invalidate_linear_stage_but_backend_does():
    X, y, ids, folds = small_data()
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2)
    params = {"lambd": 0.02, "gamma": 0.3, "retention_fraction": 0.75}
    first = evaluator.evaluate(params)
    evaluator.kernel_cache_bytes = 512
    second = evaluator.evaluate(params)
    assert evaluator.linear_fit_count_ == len(folds)
    assert all(fold["linear_cache_hit"] for fold in second["folds"])
    assert [fold["selected_ids_sha256"] for fold in first["folds"]] == [
        fold["selected_ids_sha256"] for fold in second["folds"]]
    evaluator.optimizer_backend = "auto"
    third = evaluator.evaluate(params)
    assert evaluator.linear_fit_count_ == 2 * len(folds)
    assert not any(fold["linear_cache_hit"] for fold in third["folds"])
    assert third["execution_identity"]["requested_backend"] == "auto"
    evaluator.optimizer_backend = "python"
    fourth = evaluator.evaluate(params)
    assert evaluator.linear_fit_count_ == 3 * len(folds)
    assert fourth["mean_score"] == first["mean_score"]


@pytest.mark.parametrize("changed_field, changed_value", [
    ("extension_available", False), ("extension_sha256", "b" * 64),
])
def test_changed_native_identity_cannot_reuse_a_cached_linear_model(changed_field, changed_value, monkeypatch):
    X, y, ids, folds = small_data()
    identity = {"requested_backend": "auto", "extension_available": True,
                "extension_filename": "native.pyd", "extension_sha256": "a" * 64,
                "package_version": mnist_cv._PACKAGE_VERSION}
    monkeypatch.setattr(mnist_cv, "execution_identity", lambda backend: dict(identity))
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2, optimizer_backend="auto")
    params = {"lambd": 0.02, "gamma": 0.3, "retention_fraction": 0.75}
    evaluator.evaluate(params)
    identity[changed_field] = changed_value
    with pytest.raises(RuntimeError, match="execution identity changed"):
        evaluator.evaluate(params)
    assert evaluator.linear_fit_count_ == len(folds)


def test_identity_change_during_fit_prevents_completed_result(monkeypatch):
    X, y, ids, folds = small_data()
    identity = {"requested_backend": "auto", "extension_available": True,
                "extension_filename": "native.pyd", "extension_sha256": "a" * 64,
                "package_version": mnist_cv._PACKAGE_VERSION}
    monkeypatch.setattr(mnist_cv, "execution_identity", lambda backend: dict(identity))
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2, optimizer_backend="auto")
    original_fit = KernelSGDDWD.fit

    def changing(model, *args, **kwargs):
        result = original_fit(model, *args, **kwargs)
        identity["extension_sha256"] = "b" * 64
        return result

    monkeypatch.setattr(KernelSGDDWD, "fit", changing)
    with pytest.raises(RuntimeError, match="execution identity changed"):
        evaluator.evaluate({"lambd": 0.02, "gamma": 0.3, "retention_fraction": 0.75})


def test_unavailable_auto_reports_fallback_and_native_errors_before_fitting(monkeypatch):
    X, y, ids, folds = small_data()
    original_import = mnist_cv.importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name in ("two_stage_sgd_dwd._sgd_accel", "._sgd_accel"):
            raise ImportError("Synthetic portable installation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(mnist_cv.importlib, "import_module", unavailable)
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2,
                                      optimizer_backend="auto", kernel_cache_bytes=512)
    record = evaluator.evaluate({"lambd": 0.02, "gamma": 0.3, "retention_fraction": 0.75})
    assert record["execution_identity"]["extension_available"] is False
    for fold in record["folds"]:
        for stage in ("linear_fit", "kernel_fit"):
            assert fold[stage]["optimizer_backend"] == "python"
            assert fold[stage]["optimizer_backend_reason"] == "extension_unavailable"
    with pytest.raises(ImportError, match="optional compiled"):
        mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2, optimizer_backend="native")


def test_auto_batch_fallback_records_actual_path_without_kernel_cache():
    X, y, ids, folds = small_data()
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2,
                                      batch_size=2, optimizer_backend="auto")
    record = evaluator.evaluate({"lambd": 0.02, "gamma": 0.3, "retention_fraction": 0.75})
    for fold in record["folds"]:
        for stage in ("linear_fit", "kernel_fit"):
            assert fold[stage]["optimizer_backend"] == "python"
            assert fold[stage]["optimizer_backend_reason"] == "batch_size_not_one"


@pytest.mark.parametrize("controls", [
    {"optimizer_backend": "invalid"}, {"optimizer_backend": True},
    {"kernel_cache_bytes": True}, {"kernel_cache_bytes": -1}, {"kernel_cache_bytes": 1.5},
    {"kernel_cache_bytes": 512, "batch_size": 2},
    {"kernel_cache_bytes": 512, "batch_size": "full"},
    {"optimizer_backend": "native", "batch_size": 2},
])
def test_invalid_acceleration_controls_fail_before_any_fit(controls, monkeypatch):
    X, y, ids, folds = small_data()

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid controls reached an optimizer")

    monkeypatch.setattr(LinearSGDDWD, "fit", unexpected)
    monkeypatch.setattr(KernelSGDDWD, "fit", unexpected)
    with pytest.raises(ValueError):
        mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2, **controls)


def test_native_numerical_failure_propagates_without_python_retry(monkeypatch):
    if not mnist_cv.execution_identity("auto")["extension_available"]:
        pytest.skip("Optional native extension unavailable")
    X, y, ids, folds = small_data()
    native = mnist_cv.importlib.import_module("two_stage_sgd_dwd._sgd_accel")
    calls = []

    def failure(kind, coefficients, bias, get_row, labels, indices, rates, lambd):
        calls.append(kind)
        return coefficients.copy(), bias, 0, "Synthetic native numerical failure"

    monkeypatch.setattr(native, "run_steps", failure)
    evaluator = mnist_cv.FoldEvaluator(X, y, ids, folds, max_epochs=2, optimizer_backend="native")
    with pytest.raises(NumericalFailure, match="Synthetic native numerical failure"):
        evaluator.evaluate({"lambd": 0.02, "gamma": 0.3, "retention_fraction": 0.75})
    assert calls == [0]
    assert evaluator.linear_fit_count_ == 0
