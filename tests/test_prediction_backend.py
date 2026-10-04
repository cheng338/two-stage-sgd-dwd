"""Public prediction-backend plumbing, independent of fast-distance details."""
import copy
import importlib.util
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from two_stage_sgd_dwd import KernelSGDDWD, LinearSGDDWD, TwoStageDWDClassifier
from two_stage_sgd_dwd import _estimators
from two_stage_sgd_dwd._estimators import _scores


def linear_kernel(X, Y):
    return X @ Y.T


def data():
    values = np.array([-2.0, -1.4, -0.7, -0.2, 0.2, 0.7, 1.4, 2.0])
    X = np.column_stack((values, values**2 / 5))
    return X, np.where(values > 0, "positive", "negative")


def estimator(kind, **overrides):
    controls = dict(gamma=0.4, max_epochs=3, random_state=12)
    controls.update(overrides)
    if kind == "two_stage":
        return TwoStageDWDClassifier(retention_fraction=1.0, **controls)
    return KernelSGDDWD(**controls)


def kernel_stage(model):
    return getattr(model, "kernel_estimator_", model)


def direct_reference(model, X):
    kernel = kernel_stage(model)
    return _scores("kernel", X, kernel.dual_coef_[0], kernel.intercept_[0],
                   kernel.basis_, kernel.gamma_, kernel.prediction_batch_size,
                   kernel_spec=kernel._kernel_spec_)


@pytest.mark.parametrize("kind", ["kernel", "two_stage"])
def test_default_clone_and_unchanged_direct_scores(kind):
    X, y = data()
    model = estimator(kind)
    assert model.get_params()["prediction_backend"] == "direct"
    assert clone(model).prediction_backend == "direct"
    assert clone(model.set_params(prediction_backend="blas")).prediction_backend == "blas"
    model.set_params(prediction_backend="direct").fit(X, y)
    actual, expected = model.decision_function(X), direct_reference(model, X)
    assert actual.tobytes() == expected.tobytes()
    assert kernel_stage(model).prediction_backend == "direct"
    assert "prediction_backend" not in LinearSGDDWD().get_params()


@pytest.mark.parametrize("kind", ["kernel", "two_stage"])
def test_positive_rbf_dispatch_and_postfit_switch_do_not_mutate_nested_state(kind, monkeypatch):
    X, y = data()
    model = estimator(kind).fit(X, y)
    direct = model.decision_function(X)
    seen = []
    expected = np.linspace(-0.75, 0.75, len(X))

    def fast(query, basis, coefficients, intercept, gamma, batch_size):
        seen.append((query, basis, coefficients, intercept, gamma, batch_size))
        assert_array_equal(query, X)
        assert basis is kernel_stage(model).basis_
        assert np.shares_memory(coefficients, kernel_stage(model).dual_coef_)
        return expected.copy()

    monkeypatch.setattr(_estimators, "rbf_prediction_scores_blas", fast)
    model.set_params(prediction_backend="blas")
    before = pickle.dumps(model)
    assert_array_equal(model.decision_function(X), expected)
    assert_array_equal(model.predict(X), model.classes_[(expected >= 0).astype(int)])
    assert model.score(X, y) == np.mean(model.classes_[(expected >= 0).astype(int)] == y)
    assert pickle.dumps(model) == before
    if kind == "two_stage":
        assert model.kernel_estimator_.prediction_backend == "direct"
    assert len(seen) == 3
    model.set_params(prediction_backend="direct")
    assert model.decision_function(X).tobytes() == direct.tobytes()
    assert len(seen) == 3


@pytest.mark.parametrize("kind", ["kernel", "two_stage"])
@pytest.mark.parametrize("batch_size", [1, 2, "full"])
def test_fast_prediction_is_excluded_from_training_checkpoint_and_norm(kind, batch_size, monkeypatch):
    X, y = data()
    callbacks = []

    def unexpected(*args, **kwargs):
        raise AssertionError("The fast prediction helper reached training or checkpoint scoring")

    monkeypatch.setattr(_estimators, "rbf_prediction_scores_blas", unexpected)
    controls = dict(batch_size=batch_size, validation_interval=1, min_epochs=0,
                    stopping="validation")
    direct = estimator(kind, **controls).fit(X, y, validation_data=(X, y))
    accelerated = estimator(kind, prediction_backend="blas", **controls).fit(
        X, y, validation_data=(X, y), checkpoint_callback=lambda *args: callbacks.append(args))
    assert callbacks
    assert kernel_stage(accelerated).prediction_backend == "blas"
    pairs = [(kernel_stage(direct), kernel_stage(accelerated))]
    if kind == "two_stage":
        pairs.append((direct.linear_estimator_, accelerated.linear_estimator_))
        assert direct.stage_seeds_ == accelerated.stage_seeds_
        assert_array_equal(direct.selected_indices_, accelerated.selected_indices_)
        assert_array_equal(direct.selection_scores_, accelerated.selection_scores_)
        assert "prediction_backend" not in accelerated.linear_estimator_.get_params()
    for expected, actual in pairs:
        for field in ("initial_state_", "final_state_", "selected_state_"):
            left, right = getattr(expected, field), getattr(actual, field)
            assert_array_equal(left["coefficients"], right["coefficients"])
            assert left["intercept"] == right["intercept"]
            assert left["iteration"] == right["iteration"]
        assert expected.rng_state_ == actual.rng_state_
        assert expected.stop_reason_ == actual.stop_reason_
        assert expected.selected_iteration_ == actual.selected_iteration_
        assert [row["validation_loss"] for row in expected.history_] == [
            row["validation_loss"] for row in actual.history_]
    assert kernel_stage(accelerated).rkhs_norm_squared() == kernel_stage(direct).rkhs_norm_squared()


@pytest.mark.parametrize("kind", ["kernel", "two_stage"])
@pytest.mark.parametrize("settings", [dict(kernel="linear"), dict(kernel="poly"),
                                      dict(kernel=linear_kernel), dict(gamma=0.0)])
def test_other_kernel_and_zero_width_routes_keep_direct_behavior(kind, settings, monkeypatch):
    X, y = data()
    model = estimator(kind, prediction_backend="blas", **settings).fit(X, y)

    def unexpected(*args, **kwargs):
        raise AssertionError("Nonpositive RBF or another kernel reached fast RBF prediction")

    monkeypatch.setattr(_estimators, "rbf_prediction_scores_blas", unexpected)
    assert model.decision_function(X).tobytes() == direct_reference(model, X).tobytes()


@pytest.mark.parametrize("kind", ["kernel", "two_stage"])
@pytest.mark.parametrize("invalid", [None, True, 1, "auto", "native", "BLAS", ["blas"]])
def test_invalid_backend_is_rejected_at_fit_and_after_fitting(kind, invalid, monkeypatch):
    X, y = data()
    fitted = estimator(kind).fit(X, y)
    fitted.prediction_backend = invalid
    with pytest.raises(ValueError, match="prediction_backend"):
        fitted.decision_function(X)

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid prediction controls reached optimization")

    monkeypatch.setattr(_estimators, "_optimize", unexpected)
    with pytest.raises(ValueError, match="prediction_backend"):
        estimator(kind, prediction_backend=invalid).fit(X, y)


@pytest.mark.parametrize("kind", ["kernel", "two_stage"])
def test_old_serialized_models_default_to_direct_without_state_mutation(kind, monkeypatch):
    X, y = data()
    model = estimator(kind).fit(X, y)
    expected = model.decision_function(X)
    del model.prediction_backend
    if kind == "two_stage":
        del model.kernel_estimator_.prediction_backend
    restored = pickle.loads(pickle.dumps(model))
    before = pickle.dumps(restored)

    def unexpected(*args, **kwargs):
        raise AssertionError("An old serialized model selected fast prediction")

    monkeypatch.setattr(_estimators, "rbf_prediction_scores_blas", unexpected)
    assert restored.decision_function(X).tobytes() == expected.tobytes()
    assert pickle.dumps(restored) == before
    assert not hasattr(restored, "prediction_backend")


@pytest.mark.parametrize("kind", ["kernel", "two_stage"])
def test_public_fittedness_and_feature_validation_stay_in_place(kind, monkeypatch):
    X, y = data()
    model = estimator(kind, prediction_backend="blas")
    with pytest.raises(NotFittedError):
        model.decision_function(X)
    model.fit(X, y)

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid public features reached fast scoring")

    monkeypatch.setattr(_estimators, "rbf_prediction_scores_blas", unexpected)
    for invalid in (np.empty((0, 2)), np.ones((2, 3)), np.full((2, 2), np.nan),
                    np.full((2, 2), np.inf), np.ones(2)):
        with pytest.raises(ValueError):
            model.decision_function(invalid)
    if kind == "two_stage":
        model.kernel_estimator_._fitted_ = False
        with pytest.raises(NotFittedError):
            model.decision_function(X)


@pytest.mark.parametrize("kind", ["kernel", "two_stage"])
def test_fast_prediction_runs_without_native_optimizer_import_and_roundtrips(kind, monkeypatch):
    X, y = data()
    model = estimator(kind, prediction_backend="blas", optimizer_backend="python").fit(X, y)
    from two_stage_sgd_dwd import _native_optimizer

    def unexpected(*args, **kwargs):
        raise AssertionError("Prediction attempted to import a native optimizer")

    monkeypatch.setattr(_native_optimizer.importlib, "import_module", unexpected)
    before = pickle.dumps(model)
    actual = model.decision_function(X)
    assert_allclose(actual, direct_reference(model, X), rtol=1e-12, atol=1e-14)
    assert_array_equal(model.predict(X), model.classes_[(actual >= 0).astype(int)])
    assert pickle.dumps(model) == before
    assert_array_equal(pickle.loads(before).decision_function(X), actual)


@pytest.mark.parametrize("kind", ["kernel", "two_stage"])
def test_native_fitting_does_not_call_fast_prediction(kind, monkeypatch):
    if importlib.util.find_spec("two_stage_sgd_dwd._sgd_accel") is None:
        pytest.skip("Optional native optimizer unavailable")
    X, y = data()

    def unexpected(*args, **kwargs):
        raise AssertionError("Native fitting called public fast prediction")

    monkeypatch.setattr(_estimators, "rbf_prediction_scores_blas", unexpected)
    direct = estimator(kind, optimizer_backend="native", kernel_cache_bytes=256).fit(X, y)
    fast = estimator(kind, optimizer_backend="native", kernel_cache_bytes=256,
                     prediction_backend="blas").fit(X, y)
    assert_array_equal(kernel_stage(direct).dual_coef_, kernel_stage(fast).dual_coef_)
    assert_array_equal(kernel_stage(direct).intercept_, kernel_stage(fast).intercept_)
    assert kernel_stage(direct).rng_state_ == kernel_stage(fast).rng_state_
    if kind == "two_stage":
        assert_array_equal(direct.selected_indices_, fast.selected_indices_)
        assert_array_equal(direct.linear_estimator_.coef_, fast.linear_estimator_.coef_)
