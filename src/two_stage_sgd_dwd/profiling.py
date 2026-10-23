"""Opt-in diagnostics for the bounded RBF prediction helpers.

Profiles contain scalar counters and timings only; they never retain features,
models, kernels, or predictions. Profiling does not change numerical settings.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import get_ident
from time import perf_counter


_active_profile = ContextVar("sgd_dwd_prediction_profile", default=None)
_active_call = ContextVar("sgd_dwd_prediction_call", default=None)


def _owner():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return get_ident(), task


class _Call:
    def __init__(self, records, backend, parent):
        self.records = records
        self.record = {
            "call_id": len(records), "parent_call_id": parent,
            "backend": backend, "status": "running", "error_type": None,
            "blocks_started": 0, "blocks_completed": 0, "rows_completed": 0,
            "fallbacks": {},
            "seconds": dict.fromkeys(("setup", "kernel", "score", "guard", "delegated"), 0.0),
        }
        records.append(self.record)
        self._phase = "setup"
        self._last = self._start = perf_counter()

    def phase(self, name):
        now = perf_counter()
        self.record["seconds"][self._phase] += now - self._last
        self._last, self._phase = now, name

    def start_block(self):
        self.record["blocks_started"] += 1
        self.phase("kernel")

    def complete_block(self, rows):
        self.record["blocks_completed"] += 1
        self.record["rows_completed"] += int(rows)

    def fallback(self, reason, rows, triggered_rows=None):
        event = self.record["fallbacks"].setdefault(reason, {
            "events": 0, "events_completed": 0,
            "rows_requested": 0, "rows_completed": 0,
            "triggered_rows": 0,
        })
        event["events"] += 1
        event["rows_requested"] += int(rows)
        event["triggered_rows"] += int(rows if triggered_rows is None else triggered_rows)

    def complete_fallback(self, reason, rows):
        event = self.record["fallbacks"][reason]
        event["events_completed"] += 1
        event["rows_completed"] += int(rows)

    def finish(self, error):
        self.phase(self._phase)
        self.record["seconds"]["total"] = self._last - self._start
        self.record["status"] = "ok" if error is None else "failed"
        self.record["error_type"] = error


@contextmanager
def prediction_profile():
    """Yield a list of records for RBF helper calls in this thread/task.

    Example::

        with prediction_profile() as records:
            scores = model.decision_function(X)

    Nested contexts collect separately and restore the enclosing context on
    exit. Other threads and child asyncio tasks must open their own context.
    Linear, callable, and precomputed kernels are not instrumented. A direct
    fallback of a BLAS helper creates a child record; its time is also reported
    as ``delegated`` in the parent, so parent/child totals must not be added.

    ``kernel`` includes distance computation and exponentiation; ``score`` is
    score accumulation and its finite checks; ``guard`` includes uncertainty
    checks. Fallback counters record requested and successfully completed work
    (distance rows for ``distance_guard``, direct-distance and
    scaled-exponent checks for ``near_constant_probe``, full scores for other
    reasons). A completed probe need not lead to a ``near_constant_score``
    correction. ``execution_direct`` counts the first evaluation of a small
    block by direct distances, not a recomputation. ``triggered_rows`` distinguishes uncertain scores from the full
    blocks recomputed for them. Failed calls retain partial progress. Timings
    include profiling overhead and are intended for diagnosis, not benchmarking.
    """
    records = []
    session = {"owner": _owner(), "records": records, "active": True}
    token = _active_profile.set(session)
    try:
        yield records
    finally:
        # A copied Context can outlive this scope on the same thread/task.
        # Revoke it as well as restoring the current ContextVar binding.
        session["active"] = False
        _active_profile.reset(token)


def _profiled(backend):
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            active = _active_profile.get()
            if active is None or not active["active"] or active["owner"] != _owner():
                return function(*args, **kwargs)
            parent = _active_call.get()
            # A nested profile context owns a distinct list, even if opened
            # during an outer helper call.
            parent_id = (parent.record["call_id"] if parent is not None and
                         parent.records is active["records"] else None)
            call = _Call(active["records"], backend, parent_id)
            token = _active_call.set(call)
            error = None
            try:
                return function(*args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                try:
                    call.finish(error)
                finally:
                    _active_call.reset(token)
        return run
    return decorate


def _current_call():
    active = _active_profile.get()
    call = _active_call.get()
    # The owner check also covers explicitly copied contexts in another thread.
    if (active is None or not active["active"] or call is None or
            active["owner"] != _owner()):
        return None
    return call if call.records is active["records"] else None
