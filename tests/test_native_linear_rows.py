"""Tuple row access preserves native scalar updates and ownership semantics."""
import gc
import sys
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from two_stage_sgd_dwd import KernelSGDDWD, LinearSGDDWD, _native_optimizer
from test_kernel_cache import ExactAssertions, problem

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None


HAS_TUPLE_ROWS = bool(getattr(_sgd_accel, "_supports_tuple_rows", False))
requires_tuple_rows = unittest.skipUnless(HAS_TUPLE_ROWS, "Tuple-row native capability unavailable")


def unaligned_row():
    result = np.ndarray((2,), dtype=np.float64, buffer=bytearray(17), offset=1)
    result[:] = [1., 2.]
    return result


@requires_tuple_rows
class NativeTupleRowTests(ExactAssertions):
    def run_rows(self, rows, indices, rates=None, initial=None, labels=None, bias=0., kind=0):
        indices = np.asarray(indices, dtype=np.int64)
        if rates is None:
            rates = np.full(len(indices), .1)
        if initial is None:
            initial = np.zeros(2)
        if labels is None:
            labels = np.array([-1., 1.])
        return _sgd_accel.run_steps(kind, initial, bias, rows, labels, indices,
                                    np.asarray(rates, dtype=np.float64), .2)

    def every_prefix(self, rows, indices, rates, initial, labels, bias=0.):
        saved = [row.tobytes() for row in rows]
        initial_bytes = initial.tobytes()
        for stop in range(len(indices) + 1):
            with self.subTest(width=len(initial), prefix=stop):
                args = (indices[:stop], rates[:stop], initial, labels, bias)
                with np.errstate(all="ignore"):
                    expected = self.run_rows(rows.__getitem__, *args)
                    actual = self.run_rows(rows, *args)
                self.assertExact(actual, expected)
                self.assertEqual(initial.tobytes(), initial_bytes)
                self.assertEqual([row.tobytes() for row in rows], saved)
                self.assertFalse(np.shares_memory(actual[0], initial))
                for row in rows:
                    self.assertFalse(np.shares_memory(actual[0], row))

    def test_random_and_projection_trajectories_match_at_every_prefix(self):
        for seed in (17, 603):
            for width in (1, 2, 7, 33, 128):
                rng = np.random.default_rng(seed)
                rows = tuple(np.array(row, copy=True) for row in rng.normal(size=(5, width)))
                labels = np.array([-1., 1., -1., 1., 1.])
                indices = rng.integers(5, size=13, dtype=np.int64)
                rates = rng.uniform(.001, .8, size=13)
                rates[0] = 0.
                for scale in (.01, 4.):
                    with self.subTest(seed=seed, width=width, scale=scale):
                        self.every_prefix(rows, indices, rates, rng.normal(size=width) * scale,
                                          labels, -.17)

    def test_finite_extremes_signed_zero_and_projection_boundaries(self):
        initials = (np.array([.6, .8]), np.array([0., -0.]),
                    np.array([1e300, -2e300]), np.array([1e-300, -1e-300]),
                    np.array([np.nextafter(1., 0.), 0.]),
                    np.array([np.nextafter(1., 2.), 0.]))
        rows = (np.zeros(2), np.array([np.nextafter(0., 1.), -0.]))
        for initial in initials:
            with self.subTest(initial=initial.tolist()):
                self.every_prefix(rows, np.array([0, 1, 0, 1], dtype=np.int64),
                                  np.array([0., .1, .2, .3]), initial, np.array([-1., 1.]))

    def test_numerical_failure_keeps_identical_committed_prefix(self):
        cases = ((np.array([.2, .4]), np.array([0., -1e308])),
                 (np.array([.2, .4]), np.array([np.nan, 0.])),
                 (np.array([.2, .4]), np.array([np.inf, 0.])))
        for rows in cases:
            with self.subTest(second=rows[1].tolist()):
                self.every_prefix(rows, np.array([0, 1, 0], dtype=np.int64),
                                  np.array([.1, 1e308, .1]), np.zeros(2), np.array([-1., -1.]))
                with np.errstate(all="ignore"):
                    actual = self.run_rows(rows, [0, 1, 0], [.1, 1e308, .1], labels=np.array([-1., -1.]))
                    prefix = self.run_rows(rows, [0], [.1], labels=np.array([-1., -1.]))
                self.assertEqual(actual[2], 1)
                self.assertIn("Nonfinite", actual[3])
                self.assertExact(actual[:2], prefix[:2])

    def compare_error(self, rows, indices, **kwargs):
        errors = []
        for access in (rows.__getitem__, rows):
            try:
                self.run_rows(access, indices, **kwargs)
            except Exception as exc:
                errors.append((type(exc), str(exc)))
            else:
                self.fail("Invalid row access unexpectedly succeeded")
        self.assertEqual(errors[0], errors[1])
        return errors[0]

    def test_indices_checked_before_access_and_tuple_length_is_lazy(self):
        for index in (-1, 2):
            with self.subTest(index=index):
                error = self.compare_error((), [index])
                self.assertIs(error[0], ValueError)
                self.assertIn("sampled observation", error[1])
        short = (np.array([1., 2.]),)
        self.assertEqual(self.run_rows(short, [0])[2], 1)
        self.assertIs(self.compare_error(short, [0, 1])[0], IndexError)
        self.assertEqual(self.run_rows((), [])[2], 0)
        # Surplus rows are not touched; fitting indices are bounded by labels.
        self.assertEqual(self.run_rows(short + (np.ones(2), object()), [0, 1])[2], 2)

    def test_invalid_unsampled_rows_and_layouts_keep_lazy_validation(self):
        bad_rows = (object(), [1., 2.], np.ones(3), np.ones(2, dtype=np.float32),
                    np.ones((1, 2)), np.arange(4., dtype=np.float64)[::2], unaligned_row())
        for bad in bad_rows:
            with self.subTest(type=type(bad).__name__, shape=getattr(bad, "shape", None)):
                rows = (np.array([.2, .3]), bad)
                self.assertEqual(self.run_rows(rows, [0])[2], 1)
                self.compare_error(rows, [0, 1])

    def test_readonly_rows_are_accepted_without_copy_or_mutation(self):
        rows = (np.array([.2, -.3]), np.array([1., .5]))
        for row in rows:
            row.setflags(write=False)
        self.every_prefix(rows, np.array([0, 1, 0], dtype=np.int64), np.array([.2, .4, .7]),
                          np.array([.1, -.1]), np.array([-1., 1.]))
        self.assertTrue(all(not row.flags.writeable for row in rows))

    def test_references_are_balanced_on_success_numeric_and_structural_failure(self):
        cases = ((np.ones(2), np.zeros(2)),
                 (np.ones(2), np.array([np.nan, 0.])),
                 (np.ones(2), np.ones(3)))
        for rows in cases:
            with self.subTest(second=rows[1].tolist()):
                before = tuple(sys.getrefcount(row) for row in rows)
                for _ in range(8):
                    try:
                        with np.errstate(all="ignore"):
                            self.run_rows(rows, [0, 1, 0])
                    except ValueError:
                        pass
                self.assertEqual(tuple(sys.getrefcount(row) for row in rows), before)

    def test_result_does_not_retain_temporary_rows_or_tuple(self):
        def temporary_fit():
            rows = (np.array([.2, -.3]), np.array([1., .5]))
            references = tuple(weakref.ref(row) for row in rows)
            result = self.run_rows(rows, [0, 1, 0])
            return result, references

        result, references = temporary_fit()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertEqual(result[2], 3)
        self.assertTrue(np.isfinite(result[0]).all())

    def test_kernel_and_noncallable_tuple_subclass_retain_rejection(self):
        class TupleSubclass(tuple):
            pass

        rows = (np.array([1., .2]), np.array([.2, 1.]))
        for supplied, kind in ((rows, 1), (TupleSubclass(rows), 0), (list(rows), 0)):
            with self.subTest(kind=kind, type=type(supplied).__name__):
                with self.assertRaisesRegex(ValueError, "scalar solver controls"):
                    self.run_rows(supplied, [0], kind=kind)

    def test_callable_tuple_subclass_and_custom_callable_keep_dispatch(self):
        rows = (np.array([1., .2]), np.array([.2, 1.]))

        class CallableTuple(tuple):
            def __call__(self, index):
                calls.append((type(index), index))
                return tuple.__getitem__(self, 1 - index)

        class CallableRows:
            def __call__(self, index):
                calls.append((type(index), index))
                return rows[1 - index]

        expected = self.run_rows(lambda i: rows[1 - i], [0, 1, 0])
        for supplied in (CallableTuple(rows), CallableRows()):
            calls = []
            self.assertExact(self.run_rows(supplied, [0, 1, 0]), expected)
            self.assertEqual(calls, [(int, 0), (int, 1), (int, 0)])

    def test_structural_callback_exception_identity_is_preserved(self):
        error = RuntimeError("row callback failed")

        def fail(index):
            raise error

        with self.assertRaises(RuntimeError) as caught:
            self.run_rows(fail, [0])
        self.assertIs(caught.exception, error)


class LinearRowDispatchTests(ExactAssertions):
    def test_dispatch_with_capability_and_older_binary_fallback_without_extension(self):
        X = np.array([[.2, .3], [-.4, .5]])
        y = np.array([-1., 1.])
        for capability in (None, 0, 1):
            with self.subTest(capability=capability):
                seen = []
                error = RuntimeError("stop after inspecting native dispatch")

                def inspect(kind, coefficient, bias, access, labels, indices, rates, lambd):
                    seen.append(access)
                    self.assertEqual(kind, 0)
                    if capability:
                        self.assertIs(type(access), tuple)
                        self.assertEqual(len(access), len(X))
                    else:
                        self.assertTrue(callable(access))
                    for i in range(len(X)):
                        row = access[i] if capability else access(i)
                        self.assertExact(row, X[i])
                        self.assertTrue(np.shares_memory(row, X))
                    raise error

                native = SimpleNamespace(run_steps=inspect)
                if capability is not None:
                    native._supports_tuple_rows = capability
                est = SimpleNamespace(max_epochs=1, sampling="with_replacement")
                with self.assertRaises(RuntimeError) as caught:
                    _native_optimizer.optimize("linear", X, y, np.random.default_rng(71), est, native)
                self.assertIs(caught.exception, error)
                self.assertEqual(len(seen), 1)

    @requires_tuple_rows
    def test_public_fit_uses_tuple_linear_callable_kernel_and_older_binary_fallback(self):
        X, y = problem()
        original = _sgd_accel.run_steps
        for constructor in (LinearSGDDWD, KernelSGDDWD):
            for sampling in ("with_replacement", "shuffle"):
                with self.subTest(model=constructor.__name__, sampling=sampling):
                    calls = []

                    def trace(kind, coefficient, bias, access, labels, indices, rates, lambd):
                        calls.append((kind, type(access) is tuple, callable(access)))
                        return original(kind, coefficient, bias, access, labels, indices, rates, lambd)

                    settings = dict(max_epochs=2, random_state=72, sampling=sampling,
                                    optimizer_backend="native")
                    with patch.object(_sgd_accel, "run_steps", side_effect=trace):
                        actual = constructor(**settings).fit(X, y)
                    expected_call = (0, True, False) if constructor is LinearSGDDWD else (1, False, True)
                    self.assertEqual(calls, [expected_call, expected_call])
                    # An extension without the feature flag must still receive
                    # the callable API; compare complete recorded fitting state.
                    saved_flag = _sgd_accel._supports_tuple_rows
                    delattr(_sgd_accel, "_supports_tuple_rows")
                    try:
                        calls.clear()
                        with patch.object(_sgd_accel, "run_steps", side_effect=trace):
                            previous = constructor(**settings).fit(X, y)
                    finally:
                        _sgd_accel._supports_tuple_rows = saved_flag
                    self.assertTrue(all(not is_tuple and is_callable for _, is_tuple, is_callable in calls))
                    self.assertSameFit(actual, previous)
                    self.assertExact(actual.decision_function(X), previous.decision_function(X))


if __name__ == "__main__":
    unittest.main()
