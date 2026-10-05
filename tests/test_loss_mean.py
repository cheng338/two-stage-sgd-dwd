"""Keep representable DWD means finite without changing ordinary readouts."""

from fractions import Fraction
import importlib.util
from pathlib import Path
import pickle
import warnings

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from two_stage_sgd_dwd import (
    KernelSGDDWD, LinearSGDDWD, NumericalFailure, neg_dwd_loss,
)
from two_stage_sgd_dwd._core import _mean_finite_losses, dwd_loss

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None


@pytest.mark.parametrize("values", [
    np.full(32, np.finfo(float).max),
    np.array([np.finfo(float).max, 0.]),
    np.array([np.finfo(float).max, np.finfo(float).max / 2]),
])
def test_extreme_mean_matches_exact_rational_average(values):
    expected = float(sum((Fraction(float(v)) for v in values), Fraction()) / len(values))
    original = values.copy()
    with np.errstate(all="raise"):
        actual = _mean_finite_losses(values)
    assert actual == expected
    assert np.isfinite(actual)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("mode", ["ignore", "warn", "raise", "call", "log"])
def test_reduction_overflow_is_quiet_and_preserves_caller_error_policy(mode):
    events = []

    class Log:
        def write(self, message):
            events.append(message)

    handler = Log() if mode == "log" else lambda message, flag: events.append((message, flag))
    previous_handler = np.geterrcall()
    np.seterrcall(handler)
    try:
        with np.errstate(over=mode, invalid="raise", divide="raise", under="raise"):
            original_policy = np.geterr().copy()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                actual = _mean_finite_losses(np.full(32, 3.4e307))
            assert actual == 3.4e307
            assert not caught and not events
            assert np.geterr() == original_policy
            assert np.geterrcall() is handler
    finally:
        np.seterrcall(previous_handler)


@pytest.mark.parametrize("size", [1, 7, 32, 1000])
@pytest.mark.parametrize("scale", [1e-300, 1., 1e100, 1e300])
def test_ordinary_mean_keeps_numpy_result_bytes(size, scale):
    margins = np.random.default_rng(119).uniform(-1., 1., size * 2) * scale
    losses = dwd_loss(margins)[::2]
    original = losses.tobytes()
    expected = float(np.mean(losses))
    actual = _mean_finite_losses(losses)
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    assert losses.tobytes() == original


@pytest.mark.parametrize("values", [
    [], 1., [[1., 2.]], [np.nan], [np.inf], [-np.inf], [-1.], [1j], ["1"],
])
def test_invalid_loss_vectors_remain_rejected(values):
    with pytest.raises(ValueError, match="finite nonnegative"):
        _mean_finite_losses(values)


@pytest.fixture(params=["python", "native"])
def optimizer_backend(request):
    if request.param == "native" and _sgd_accel is None:
        pytest.skip("Optional compiled extension unavailable")
    return request.param


@pytest.mark.parametrize("constructor", [LinearSGDDWD, KernelSGDDWD])
def test_fitted_objective_and_scorer_keep_finite_extreme_means(constructor, optimizer_backend):
    X, y = np.array([[-1.], [1.]]), np.array([-1, 1])
    options = {"kernel": "linear"} if constructor is KernelSGDDWD else {}
    model = constructor(max_epochs=2, random_state=7,
                        optimizer_backend=optimizer_backend, **options).fit(X, y)
    assert model.optimizer_backend_ == optimizer_backend
    saved_model = pickle.dumps(model)
    penalty = model.lambd_ * model.rkhs_norm_squared() if options else 0.

    ordinary_mean = float(np.mean(dwd_loss(y * model.decision_function(X))))
    expected_objective = ordinary_mean + penalty if options else ordinary_mean
    assert np.asarray(model.objective(X, y)).tobytes() == np.asarray(expected_objective).tobytes()
    assert np.asarray(neg_dwd_loss(model, X, y)).tobytes() == np.asarray(-ordinary_mean).tobytes()

    query, labels = np.full((32, 1), -1e308), np.ones(32, dtype=int)
    losses = dwd_loss(model.decision_function(query))
    assert np.all(losses == losses[0]) and losses[0] > 1e307
    with np.errstate(over="raise"):
        assert model.objective(query, labels) == float(losses[0]) + penalty
        assert neg_dwd_loss(model, query, labels) == -float(losses[0])
    assert pickle.dumps(model) == saved_model


@pytest.mark.parametrize("constructor", [LinearSGDDWD, KernelSGDDWD])
@pytest.mark.parametrize("backend", ["python", "auto"])
def test_validation_accepts_representable_loss_history(constructor, backend):
    X, y = np.array([[-1.], [1.]]), np.array([-1, 1])
    query, labels = np.full((32, 1), -1e308), np.ones(32, dtype=int)
    options = {"kernel": "linear"} if constructor is KernelSGDDWD else {}
    with np.errstate(over="raise"):
        model = constructor(max_epochs=2, random_state=7, validation_interval=1,
                            optimizer_backend=backend, **options).fit(
                                X, y, validation_data=(query, labels))
    # Validation checkpoints use the Python optimizer, including auto dispatch.
    assert model.optimizer_backend_ == "python"
    assert model.model_status_ == "trained"
    losses = [row["validation_loss"] for row in model.history_ if row["validation_loss"] is not None]
    assert len(losses) == 2
    assert np.isfinite(losses).all() and max(losses) > 1e307
    assert model.best_validation_loss_ == min(losses)


def test_truly_nonfinite_penalized_objective_is_rejected(monkeypatch):
    X, y = np.array([[-1.], [1.]]), np.array([-1, 1])
    model = KernelSGDDWD(lambd=2., max_epochs=2, random_state=7).fit(X, y)
    # A finite norm can still produce an unrepresentable regularization term.
    monkeypatch.setattr(model, "rkhs_norm_squared", lambda: float(np.finfo(float).max))
    with pytest.raises(NumericalFailure, match="Nonfinite DWD objective"):
        model.objective(X, y)


def test_example_cv_retains_representable_fold_loss(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / "_mnist_cv.py"
    spec = importlib.util.spec_from_file_location("loss_mean_mnist_cv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    X = np.linspace(-1., 1., 64).reshape(-1, 1)
    y, ids = np.tile([-1, 1], 32), np.arange(64)
    folds = list(StratifiedKFold(2, shuffle=True, random_state=17).split(X, y))
    # Controlled scores isolate the fold reduction; this is not a fitted RBF claim.
    monkeypatch.setattr(KernelSGDDWD, "decision_function", lambda self, X: np.full(len(X), -3.4e307))
    with np.errstate(over="raise"):
        result = module.FoldEvaluator(X, y, ids, folds, max_epochs=1).evaluate(
            {"lambd": .1, "gamma": 1., "retention_fraction": 1.})
    assert result["eligible"]
    assert len(result["folds"]) == 2
    assert all(row["dwd_loss"] == 1.7e307 for row in result["folds"])
