"""Read-only profiling preserves prediction arithmetic and context isolation."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
import pickle

import numpy as np
import pytest
from _rbf_range_oracles import zero_or_one_kernel

from two_stage_sgd_dwd import KernelSGDDWD, TwoStageDWDClassifier, prediction_profile
from two_stage_sgd_dwd import _prediction, profiling


def inputs():
    rng = np.random.default_rng(734)
    return (rng.normal(size=(11, 33)), rng.normal(size=(13, 33)),
            rng.normal(size=13), 0.13, 0.2, 4)


def predict(backend="blas", args=None):
    function = (_prediction.rbf_prediction_scores_blas if backend == "blas"
                else _prediction.rbf_prediction_scores)
    return function(*(inputs() if args is None else args))


def check_record(record):
    assert record["status"] == "ok"
    assert record["error_type"] is None
    times = record["seconds"]
    assert all(value >= 0.0 for value in times.values())
    assert sum(value for key, value in times.items() if key != "total") == pytest.approx(times["total"])


@pytest.mark.parametrize("backend", ["direct", "blas"])
@pytest.mark.parametrize("batch", [1, 4, 100])
@pytest.mark.parametrize("kind", ["ordinary", "large_offset", "subnormal", "fortran"])
def test_enabled_disabled_exact_parity(backend, batch, kind):
    X, basis, coef, intercept, gamma, _ = inputs()
    if kind == "large_offset":
        X, basis = X + 1e12, basis + 1e12
    elif kind == "subnormal":
        coef = coef * 1e-320
        intercept = 0.0
    elif kind == "fortran":
        X, basis = np.asfortranarray(X), np.asfortranarray(basis)
    args = X, basis, coef, intercept, gamma, batch
    before = pickle.dumps(args)
    expected = predict(backend, args)
    with prediction_profile() as records:
        actual = predict(backend, args)
    assert actual.tobytes() == expected.tobytes()
    assert pickle.dumps(args) == before
    assert len(records) == 1
    check_record(records[0])
    assert records[0]["backend"] == backend
    assert records[0]["rows_completed"] == len(X)
    assert records[0]["blocks_completed"] == (len(X) + batch - 1) // batch
    # Every record is portable scalar metadata, without arrays or model state.
    json.dumps(records, allow_nan=False)


def test_no_timer_reads_without_profile(monkeypatch):
    def unexpected():
        raise AssertionError("Profiling clock read while disabled")
    monkeypatch.setattr(profiling, "perf_counter", unexpected)
    predict("direct")
    predict("blas")
    args = list(inputs())
    args[2] = args[2].astype(np.float32)
    predict("blas", args)


@pytest.mark.parametrize("cls", [KernelSGDDWD, TwoStageDWDClassifier])
@pytest.mark.parametrize("backend", ["direct", "blas"])
def test_fitted_scores_labels_and_serialization_unchanged(cls, backend):
    X, _, _, _, _, _ = inputs()
    y = np.where(np.arange(len(X)) % 2, "a", "b")
    controls = dict(max_epochs=2, random_state=7, prediction_backend=backend)
    if cls is TwoStageDWDClassifier:
        controls["retention_fraction"] = 1.0
    model = cls(**controls).fit(X, y)
    query = X + 0.03
    expected, labels = model.decision_function(query), model.predict(query)
    before = pickle.dumps(model)
    with prediction_profile() as records:
        actual = model.decision_function(query)
        actual_labels = model.predict(query)
    assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(actual_labels, labels)
    assert pickle.dumps(model) == before
    assert len(records) == 2
    records_before = pickle.dumps(records)
    model.predict(query)
    assert pickle.dumps(records) == records_before


def test_distance_fallback_counts_requested_and_completed_rows():
    args = list(inputs())
    args[0], args[1] = args[0] + 1e12, args[1] + 1e12
    with prediction_profile() as records:
        predict(args=args)
    event = records[0]["fallbacks"]["distance_guard"]
    assert event == dict(events=3, events_completed=3, rows_requested=11,
                         rows_completed=11, triggered_rows=11)


def test_uncertain_score_counts_full_block_and_triggering_rows():
    args = list(inputs())
    args[3] = 0.0
    args[3] = -predict("direct", args)[0]
    expected = predict(args=args)
    with prediction_profile() as records:
        actual = predict(args=args)
    assert actual.tobytes() == expected.tobytes()
    event = records[0]["fallbacks"]["uncertain_score"]
    assert event["events"] == event["events_completed"] == 1
    assert event["rows_requested"] == event["rows_completed"] == 4
    assert event["triggered_rows"] == 1


@pytest.mark.parametrize("kind,reason", [("coefficient", "unusual_fitted_state"),
                                         ("norm", "nonfinite_norm_or_mass")])
def test_whole_call_fallback_has_parent_identity(kind, reason):
    args = list(inputs())
    if kind == "coefficient":
        args[2] = args[2].astype(np.float32)
    else:
        args[0][:] = args[1][:] = 1e154
    expected = predict(args=args)
    with prediction_profile() as records:
        actual = predict(args=args)
    assert actual.tobytes() == expected.tobytes()
    assert len(records) == 2
    assert records[0]["parent_call_id"] is None
    assert records[1]["parent_call_id"] == records[0]["call_id"] == 0
    assert records[1]["backend"] == "direct"
    assert records[0]["fallbacks"][reason]["rows_completed"] == 11
    assert records[0]["seconds"]["delegated"] >= records[1]["seconds"]["total"]
    for record in records:
        check_record(record)


def test_arithmetic_retry_counts_successful_recomputation(monkeypatch):
    original = _prediction._distance_rows_to_kernel
    calls = 0

    def once(distances, width):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FloatingPointError("Injected fast intermediate overflow")
        return original(distances, width)

    monkeypatch.setattr(_prediction, "_distance_rows_to_kernel", once)
    with prediction_profile() as records:
        result = predict()
    assert np.isfinite(result).all()
    event = records[0]["fallbacks"]["arithmetic_retry"]
    assert event["events"] == event["events_completed"] == 1
    assert event["rows_requested"] == event["rows_completed"] == 4
    assert records[0]["blocks_completed"] == 3


@pytest.mark.parametrize("backend", ["direct", "blas"])
@pytest.mark.parametrize("kind", ["validation", "finite_range"])
def test_range_recovery_and_invalid_call_profiling(backend, kind):
    args = list(inputs())
    if kind == "validation":
        args[4] = -1.0
    else:
        args[4] = 1e308
        expected_kernel = zero_or_one_kernel(args[0], args[1], args[4])
        np.testing.assert_array_equal(expected_kernel, np.zeros_like(expected_kernel))
        expected = np.full(len(args[0]), args[3])
        assert predict(backend, args).tobytes() == expected.tobytes()
        with prediction_profile() as records:
            actual = predict(backend, args)
        assert actual.tobytes() == expected.tobytes()
        assert len(records) == 1
        check_record(records[0])
        assert records[0]["blocks_completed"] == 3
        assert records[0]["rows_completed"] == 11
        assert profiling._active_call.get() is None
        assert profiling._active_profile.get() is None
        return
    with pytest.raises((ValueError, FloatingPointError)) as expected:
        predict(backend, args)
    with prediction_profile() as records:
        with pytest.raises(type(expected.value), match=str(expected.value)):
            predict(backend, args)
    record = records[0]
    assert record["status"] == "failed"
    assert record["error_type"] == type(expected.value).__name__
    assert record["blocks_started"] == 0
    assert record["blocks_completed"] == record["rows_completed"] == 0
    assert profiling._active_call.get() is None
    assert profiling._active_profile.get() is None


def test_empty_query_is_successful_without_blocks():
    args = list(inputs())
    args[0] = args[0][:0]
    with prediction_profile() as records:
        assert predict(args=args).shape == (0,)
    assert records[0]["blocks_started"] == records[0]["blocks_completed"] == 0
    check_record(records[0])


def test_nested_contexts_restore_outer_and_do_not_collect_after_exit():
    with prediction_profile() as outer:
        predict()
        with prediction_profile() as inner:
            predict("direct")
        predict()
    predict()
    assert [record["backend"] for record in outer] == ["blas", "blas"]
    assert [record["backend"] for record in inner] == ["direct"]
    assert all(record["parent_call_id"] is None for record in outer + inner)


def test_copied_context_is_revoked_after_original_scope_exit(monkeypatch):
    with prediction_profile() as records:
        context = copy_context()
        context.run(predict)
    assert len(records) == 1
    before = pickle.dumps(records)

    def unexpected():
        raise AssertionError("Expired profile read the clock")

    monkeypatch.setattr(profiling, "perf_counter", unexpected)
    context.run(predict)
    assert pickle.dumps(records) == before


def test_context_is_restored_if_final_diagnostic_clock_fails(monkeypatch):
    original = profiling._Call.finish

    def fail(*args):
        raise RuntimeError("Injected diagnostic failure")

    with prediction_profile():
        monkeypatch.setattr(profiling._Call, "finish", fail)
        with pytest.raises(RuntimeError, match="Injected diagnostic failure"):
            predict()
        assert profiling._active_call.get() is None
        monkeypatch.setattr(profiling._Call, "finish", original)
        predict()
    assert profiling._active_profile.get() is None


def test_thread_contexts_are_independent_even_when_copied():
    def own_context(backend):
        with prediction_profile() as records:
            predict(backend)
        return records

    with ThreadPoolExecutor(max_workers=2) as pool:
        with prediction_profile() as outer:
            context = copy_context()
            pool.submit(context.run, predict).result()
            tasks = [pool.submit(own_context, backend) for backend in ("direct", "blas")]
            records = [task.result() for task in tasks]
            predict()
    assert len(outer) == 1
    assert [[item["backend"] for item in rows] for rows in records] == [["direct"], ["blas"]]


def test_async_child_tasks_do_not_pollute_inherited_profile():
    async def own_context(backend):
        with prediction_profile() as records:
            await asyncio.sleep(0)
            predict(backend)
        return records

    async def inherited():
        predict()

    async def run():
        with prediction_profile() as outer:
            await asyncio.create_task(inherited())
            direct, blas = await asyncio.gather(own_context("direct"), own_context("blas"))
            predict()
        return outer, direct, blas

    loop = asyncio.SelectorEventLoop()
    try:
        outer, direct, blas = loop.run_until_complete(run())
    finally:
        loop.close()
    assert len(outer) == len(direct) == len(blas) == 1
    assert [outer[0]["backend"], direct[0]["backend"], blas[0]["backend"]] == ["blas", "direct", "blas"]
