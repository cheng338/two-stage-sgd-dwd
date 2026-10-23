"""Tiny RBF variations survive readout without altering fitted updates."""
import pickle

import numpy as np
import pytest

from two_stage_sgd_dwd import KernelSGDDWD, TwoStageDWDClassifier, prediction_profile
from two_stage_sgd_dwd import _prediction, profiling
from two_stage_sgd_dwd._estimators import _scores


@pytest.mark.parametrize("kind", [KernelSGDDWD, TwoStageDWDClassifier])
@pytest.mark.parametrize("gamma", [1e-12, 1e-30])
@pytest.mark.parametrize("offset", [0.0, 1e9])
@pytest.mark.parametrize("batch", [1, 2, 256])
def test_genuine_symmetric_fits_preserve_the_sign_of_the_rbf_function(kind, gamma, offset, batch):
    positive = np.linspace(0.1, 3.0, 3)[:, None]
    X = np.concatenate((-positive, positive)) + offset
    y = np.array([-1, -1, -1, 1, 1, 1])
    controls = dict(gamma=gamma, lambd=1e-12, max_epochs=30, batch_size="full",
                    prediction_batch_size=batch, random_state=3)
    if kind is TwoStageDWDClassifier:
        controls["retention_fraction"] = 1.0
    model = kind(**controls).fit(X, y)
    stage = getattr(model, "kernel_estimator_", model)
    order = np.argsort(stage.basis_[:, 0])
    basis, coefficients = stage.basis_[order, 0], stage.dual_coef_[0, order]
    # Each positive coefficient multiplies k(x,+t)-k(x,-t), whose sign is x.
    # This independent identity remains valid even when exp rounds both to1.
    np.testing.assert_array_equal(basis[:3] - offset, -(basis[3:] - offset)[::-1])
    np.testing.assert_array_equal(coefficients[:3], -coefficients[3:][::-1])
    assert np.all(coefficients[3:] > 0) and stage.intercept_[0] == 0
    query = np.array([[-1.0], [-1e-6], [0.0], [1e-6], [1.0]]) + offset
    before = pickle.dumps(model)
    scores = model.decision_function(query)
    np.testing.assert_array_equal(np.sign(scores), [-1, -1, 0, 1, 1])
    np.testing.assert_array_equal(model.predict(query), [-1, -1, 1, 1, 1])
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("kind", [KernelSGDDWD, TwoStageDWDClassifier])
def test_explicit_direct_backend_keeps_its_original_formula(kind, monkeypatch):
    X = np.array([[-3.], [-1.], [1.], [3.]])
    controls = dict(gamma=1e-30, max_epochs=5, batch_size="full", random_state=3,
                    prediction_backend="direct")
    if kind is TwoStageDWDClassifier:
        controls["retention_fraction"] = 1.0
    model = kind(**controls).fit(X, [-1, -1, 1, 1])
    stage = getattr(model, "kernel_estimator_", model)
    query = np.array([[-1e-6], [0.], [1e-6]])
    expected = _scores("kernel", query, stage.dual_coef_[0], stage.intercept_[0],
                       stage.basis_, stage.gamma_, stage.prediction_batch_size,
                       kernel_spec=stage._kernel_spec_)

    def unexpected(*args):
        raise AssertionError("Explicit direct prediction reached near-constant repair")

    monkeypatch.setattr(_prediction, "_near_constant_scores", unexpected)
    assert model.decision_function(query).tobytes() == expected.tobytes()


@pytest.mark.parametrize("neighbor,eligible", [(-1, True), (0, True), (1, False)])
def test_precision_threshold_and_rejected_probe_are_reported(neighbor, eligible):
    width = np.sqrt(np.finfo(float).eps)
    if neighbor:
        width = np.nextafter(width, np.inf if neighbor > 0 else 0.0)
    with prediction_profile() as records:
        values = _prediction.rbf_prediction_scores_blas(
            np.array([[0.]]), np.array([[-1.], [1.]]), np.array([-1., 1.]), 0., width, 2)
    assert values[0] == 0
    events = records[0]["fallbacks"]
    probe = events["near_constant_probe"]
    assert probe["events"] == probe["events_completed"] == 1
    assert probe["rows_requested"] == probe["rows_completed"] == 1
    assert ("near_constant_score" in events) is eligible
    if eligible:
        assert events["near_constant_score"]["rows_completed"] == 1


def test_failed_probe_records_requested_but_not_completed_work(monkeypatch):
    original = _prediction.cdist

    def fail_probe(*args, **kwargs):
        record = profiling._current_call().record
        probe = record["fallbacks"].get("near_constant_probe", {})
        if probe.get("events", 0) > probe.get("events_completed", 0):
            raise FloatingPointError("Failed direct-distance probe")
        return original(*args, **kwargs)

    monkeypatch.setattr(_prediction, "cdist", fail_probe)
    with prediction_profile() as records:
        with pytest.raises(FloatingPointError, match="Failed direct-distance probe"):
            _prediction.rbf_prediction_scores_blas(
                np.array([[0.]]), np.array([[-1.], [1.]]), np.array([-1., 1.]), 0., 1e-30, 2)
    record = records[0]
    assert record["status"] == "failed" and record["rows_completed"] == 0
    event = record["fallbacks"]["near_constant_probe"]
    assert event["rows_requested"] == 1 and event["rows_completed"] == 0
    assert event["events"] == 1 and event["events_completed"] == 0
    assert "near_constant_score" not in record["fallbacks"]


def test_near_constant_correction_also_follows_arithmetic_retry(monkeypatch):
    original = _prediction._distance_rows_to_kernel
    calls = 0

    def retry_once(distances, width):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FloatingPointError("Injected fast-intermediate failure")
        return original(distances, width)

    monkeypatch.setattr(_prediction, "_distance_rows_to_kernel", retry_once)
    # Preserve the same one-dimensional geometry while reaching the norm/dot
    # route beyond the small direct-execution feature limit.
    query = np.pad(np.array([[-1e-6], [0.], [1e-6]]), ((0, 0), (0, 32)))
    basis = np.pad(np.array([[-1.], [1.]]), ((0, 0), (0, 32)))
    with prediction_profile() as records:
        scores = _prediction.rbf_prediction_scores_blas(
            query, basis,
            np.array([-1., 1.]), 0., 1e-30, 4)
    np.testing.assert_array_equal(np.sign(scores), [-1, 0, 1])
    events = records[0]["fallbacks"]
    assert events["arithmetic_retry"]["rows_completed"] == 3
    assert events["near_constant_probe"]["rows_completed"] == 3
    assert events["near_constant_score"]["rows_completed"] == 3


def test_partial_blocks_and_readonly_strided_coefficients_are_not_mutated():
    query = np.array([[-1.], [-1e-6], [0.], [1e-6], [1.]])
    basis = np.array([[-1.], [1.]])
    storage = np.array([-1., 9., 1., 9.])
    coefficients = storage[::2]
    for value in (query, basis, storage, coefficients):
        value.flags.writeable = False
    args = query, basis, coefficients, 0., 1e-30, 2
    before = pickle.dumps(args)
    with prediction_profile() as records:
        scores = _prediction.rbf_prediction_scores_blas(*args)
    np.testing.assert_array_equal(np.sign(scores), [-1, -1, 0, 1, 1])
    assert pickle.dumps(args) == before
    assert records[0]["blocks_completed"] == 3
    assert records[0]["fallbacks"]["near_constant_score"]["rows_completed"] == 5


def test_nonconstant_cancellation_keeps_original_score_bytes(monkeypatch):
    rng = np.random.default_rng(312)
    query, basis, coef = rng.normal(size=(9, 5)), rng.normal(size=(13, 5)), rng.normal(size=13)
    intercept = -_prediction.rbf_prediction_scores(query, basis, coef, 0., .4, 4)[0]
    args = query, basis, coef, intercept, .4, 4
    with monkeypatch.context() as context:
        context.setattr(_prediction, "_near_constant_scores", lambda *args: None)
        original = _prediction.rbf_prediction_scores_blas(*args)
    with prediction_profile() as records:
        actual = _prediction.rbf_prediction_scores_blas(*args)
    assert actual.tobytes() == original.tobytes()
    assert "near_constant_probe" not in records[0]["fallbacks"]
    assert "near_constant_score" not in records[0]["fallbacks"]


def test_approximate_kernel_rows_cannot_skip_the_independent_distance_probe():
    query, basis, coefficients = np.array([[0.]]), np.array([[-1.], [1.]]), np.array([-1., 1.])
    kernel = np.full((1, 2), 0.5)
    scores = np.array([-1e-20])
    _prediction._near_constant_scores(query, basis, coefficients, 0., 1e-30,
                                     kernel, scores, 1e-12, None, False)
    assert scores[0] == 0.
    assert np.max(np.abs(kernel)) < np.sqrt(np.finfo(float).eps)


@pytest.mark.parametrize("case", ["tiny_width", "tiny_coefficients", "coefficient_mass",
                                   "distance_overflow", "exponent_overflow", "nonfinite_intercept"])
def test_existing_finite_and_error_policy_is_preserved(case, monkeypatch):
    query, basis = np.array([[.5]]), np.array([[-1.], [-.3], [.3], [1.]])
    coef, intercept, gamma = np.array([-1., -.2, .2, 1.]), 0., 1e-12
    if case == "tiny_width":
        gamma = np.nextafter(0., 1.)
    elif case == "tiny_coefficients":
        coef *= 1e-300
    elif case == "coefficient_mass":
        query, basis = np.zeros((1, 1)), np.zeros((4, 1))
        coef = np.array([1e308, -1e308, 1e308, -1e308])
    elif case == "distance_overflow":
        basis *= 1e308
    elif case == "exponent_overflow":
        query[:] = 2.
        gamma = 1e308
    else:
        intercept = np.inf
    args = query, basis, coef, intercept, gamma, 2

    def outcome():
        try:
            with np.errstate(all="raise"):
                values = _prediction.rbf_prediction_scores_blas(*args)
            return "finite", np.isfinite(values).all()
        except (ValueError, FloatingPointError, OverflowError) as error:
            return "error", type(error)

    with monkeypatch.context() as context:
        context.setattr(_prediction, "_near_constant_scores", lambda *args: None)
        original = outcome()
    assert outcome() == original
