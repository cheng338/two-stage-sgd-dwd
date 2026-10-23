"""Exact epoch rates, compatibility fallbacks, and committed-prefix checks."""
import copy
import pickle
import warnings
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from two_stage_sgd_dwd import (
    KernelSGDDWD, LinearSGDDWD, NumericalFailure, TwoStageDWDClassifier,
)
from two_stage_sgd_dwd import _estimators, _native_optimizer

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None

requires_native = unittest.skipIf(_sgd_accel is None, 'Optional compiled extension unavailable')
LIMIT = 2**53


def original_rates(B, updates, m):
    return np.asarray([(1.0 / B) * (1.0 + (updates + j) / m) ** (-0.75)
                       for j in range(m)], dtype=np.float64)


def fake_native():
    def fill(out, updates, m, B):
        out[:] = original_rates(B, updates, m)
    return SimpleNamespace(fill_rates=Mock(side_effect=fill))


def problem(m=13):
    x = np.linspace(-1.5, 1.5, m)
    return np.column_stack((x, .13 * np.cos(np.arange(m)), x * x / 7)), np.where(x < 0, -1, 1)


class DifferentFloat(float):
    def __rtruediv__(self, other):
        return float(other) / float(self) + .125


class DifferentInt(int):
    def __add__(self, other):
        return int(self) + int(other) + 1


class EpochDispatchTests(unittest.TestCase):
    def assert_rates(self, actual, expected):
        self.assertEqual(actual.dtype, np.dtype('float64'))
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_supported_controls_use_helper_with_private_float64_buffer(self):
        native = fake_native()
        actual = _native_optimizer._epoch_rates(native, 8.000002, 17, 13)
        native.fill_rates.assert_called_once()
        out, updates, m, B = native.fill_rates.call_args.args
        self.assertIs(out, actual)
        self.assertTrue(out.flags.writeable)
        self.assertEqual((updates, m, B), (17, 13, 8.000002))
        self.assert_rates(actual, original_rates(B, updates, m))

    def test_missing_helper_uses_original_scalar_expression(self):
        for B, updates, m in ((8.000002, 0, 13), (1e308, 799 * 7, 7),
                              (8.0, LIMIT + 1, 3)):
            with self.subTest(B=B, updates=updates, m=m):
                actual = _native_optimizer._epoch_rates(SimpleNamespace(), B, updates, m)
                self.assert_rates(actual, original_rates(B, updates, m))

    def test_nonbuiltin_operands_preserve_scalar_overrides(self):
        cases = ((DifferentFloat(8.0), 3, 7), (np.float64(8.0), 3, 7),
                 (8.0, DifferentInt(3), 7), (8.0, np.int64(3), 7),
                 (8.0, 3, np.int64(7)), (True, 3, 7), (8.0, True, 7), (8.0, 3, True))
        for B, updates, m in cases:
            with self.subTest(types=(type(B).__name__, type(updates).__name__, type(m).__name__)):
                native = fake_native()
                actual = _native_optimizer._epoch_rates(native, B, updates, m)
                native.fill_rates.assert_not_called()
                self.assert_rates(actual, original_rates(B, updates, m))

    def test_integer_range_boundary_is_checked_before_conversion(self):
        for updates, m, expected_native in ((LIMIT - 1, 1, True), (LIMIT - 7, 7, True),
                                            (LIMIT - 6, 7, False), (LIMIT, 1, False),
                                            (LIMIT + 1, 3, False), (2**64 + 3, 3, False)):
            with self.subTest(updates=updates, m=m):
                native = fake_native()
                actual = _native_optimizer._epoch_rates(native, 8.0, updates, m)
                self.assertEqual(native.fill_rates.call_count, int(expected_native))
                self.assert_rates(actual, original_rates(8.0, updates, m))

    def test_huge_denominator_dispatch_is_checked_without_allocating(self):
        native = fake_native()
        with patch.object(_native_optimizer, 'range', create=True,
                          side_effect=RuntimeError('Observed scalar fallback before iteration')) as counter, \
                patch.object(_native_optimizer.np, 'empty',
                             side_effect=AssertionError('Must not allocate enormous array')):
            with self.assertRaisesRegex(RuntimeError, 'Observed scalar fallback'):
                _native_optimizer._epoch_rates(native, 8.0, 0, LIMIT)
        counter.assert_called_once_with(LIMIT)
        native.fill_rates.assert_not_called()

    def test_tiny_B_nonfinite_reciprocal_keeps_original_behavior(self):
        B = float(np.nextafter(0.0, 1.0))
        native = fake_native()
        actual = _native_optimizer._epoch_rates(native, B, 0, 7)
        native.fill_rates.assert_not_called()
        self.assert_rates(actual, original_rates(B, 0, 7))
        self.assertTrue(np.isposinf(actual).all())

    def test_empty_ranges_do_not_evaluate_bound_or_counter(self):
        class ForbiddenBound:
            def __rtruediv__(self, value):
                raise AssertionError('Empty range evaluated the reciprocal')

        class ForbiddenCounter:
            def __add__(self, value):
                raise AssertionError('Empty range evaluated the counter')

        for m in (0, -1, np.int64(0)):
            for B in (0.0, -0.0, ForbiddenBound()):
                for updates in (0, ForbiddenCounter()):
                    with self.subTest(m=m, bound=type(B).__name__, counter=type(updates).__name__):
                        native = fake_native()
                        self.assert_rates(_native_optimizer._epoch_rates(native, B, updates, m),
                                          original_rates(B, updates, m))
                        native.fill_rates.assert_not_called()

    def test_custom_reciprocal_failure_occurs_at_original_iteration(self):
        def evaluate(candidate):
            events = []
            error = RuntimeError('third reciprocal failed')

            class Bound(float):
                def __rtruediv__(self, value):
                    events.append(value)
                    if len(events) == 3:
                        raise error
                    return value / float(self)

            native = fake_native()
            with self.assertRaises(RuntimeError) as caught:
                if candidate:
                    _native_optimizer._epoch_rates(native, Bound(8.0), 0, 5)
                else:
                    original_rates(Bound(8.0), 0, 5)
            self.assertIs(caught.exception, error)
            native.fill_rates.assert_not_called()
            return events

        self.assertEqual(evaluate(True), evaluate(False))

    def test_original_scalar_exceptions_are_not_replaced_by_native_errors(self):
        for B, updates, m in ((0.0, 0, 3), (-0.0, 0, 3),
                              (8.0, 10**1000, 3), (8.0, -3, 3),
                              (8.0, 0, 1.5)):
            with self.subTest(B=B, updates=updates, m=m):
                native = fake_native()
                try:
                    original_rates(B, updates, m)
                except Exception as expected:
                    with self.assertRaises(type(expected)) as caught:
                        _native_optimizer._epoch_rates(native, B, updates, m)
                    self.assertEqual(str(caught.exception), str(expected))
                else:
                    self.fail('Control must exercise an exceptional scalar expression')
                native.fill_rates.assert_not_called()

    def test_helper_error_propagates_once_without_scalar_retry(self):
        for error in (ValueError('bad buffer'), MemoryError('allocation failed'),
                      FloatingPointError('controlled helper failure')):
            with self.subTest(error=type(error).__name__):
                native = SimpleNamespace(fill_rates=Mock(side_effect=error))
                with patch.object(_native_optimizer, 'range', create=True,
                                  side_effect=AssertionError('Unexpected scalar retry')):
                    with self.assertRaises(type(error)) as caught:
                        _native_optimizer._epoch_rates(native, 8.0, 0, 7)
                self.assertIs(caught.exception, error)
                native.fill_rates.assert_called_once()


class NativeRateBufferTests(unittest.TestCase):
    @requires_native
    def test_rates_are_bitwise_exact_for_realistic_and_extreme_finite_controls(self):
        for m in (1, 2, 3, 13, 14, 725, 1933):
            for updates in (0, m, 199 * m, 799 * m, LIMIT - m):
                for B in (4.0, 8.000002, 16.03125, 1e6, 1e308,
                          float(np.finfo(np.float64).max), 1e-308):
                    with self.subTest(m=m, updates=updates, B=B):
                        out = np.empty(m, dtype=np.float64)
                        answer = _sgd_accel.fill_rates(out, updates, m, B)
                        self.assertIsNone(answer)
                        self.assertEqual(out.tobytes(), original_rates(B, updates, m).tobytes())
        out = np.empty(3)
        _sgd_accel.fill_rates(out, LIMIT - 3, 3, float(np.finfo(np.float64).max))
        self.assertTrue(np.all((out > 0) & (out < np.finfo(np.float64).tiny)))

    @requires_native
    def test_rejected_controls_do_not_modify_output(self):
        cases = [(-1, 3, 8.0), (LIMIT, 3, 8.0), (LIMIT - 2, 3, 8.0),
                 (2**100, 3, 8.0), (0, 0, 8.0), (0, -1, 8.0), (0, LIMIT, 8.0),
                 (0, 3, 0.0), (0, 3, -1.0), (0, 3, float('nan')),
                 (0, 3, float('inf')), (0, 3, float(np.nextafter(0., 1.))),
                 (True, 3, 8.0), (np.int64(0), 3, 8.0), (DifferentInt(0), 3, 8.0),
                 (0, True, 8.0), (0, np.int64(3), 8.0), (0, DifferentInt(3), 8.0),
                 (0, 3, True), (0, 3, 8), (0, 3, np.float64(8.0)),
                 (0, 3, DifferentFloat(8.0))]
        for updates, m, B in cases:
            with self.subTest(updates=updates, m=m, B=B,
                              types=(type(updates).__name__, type(m).__name__, type(B).__name__)):
                out = np.array([-0.0, 123.0, -456.0])
                before = out.tobytes()
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    _sgd_accel.fill_rates(out, updates, m, B)
                self.assertEqual(out.tobytes(), before)

    @requires_native
    def test_buffer_shape_dtype_alignment_and_writability_are_checked_before_writes(self):
        storage = np.arange(6., dtype=np.float64)
        readonly = np.arange(3., dtype=np.float64)
        readonly.flags.writeable = False
        unaligned = np.ndarray((3,), dtype=np.float64, buffer=bytearray(25), offset=1)
        unaligned[:] = [1, 2, 3]
        cases = [np.empty(0), np.arange(4.), np.arange(3., dtype=np.float32),
                 np.arange(3, dtype=np.int64), np.arange(3., dtype='>f8'),
                 np.arange(3.).reshape(1, 3), storage[::2], readonly, unaligned]
        for out in cases:
            with self.subTest(shape=out.shape, dtype=out.dtype, writable=out.flags.writeable,
                              aligned=out.flags.aligned, contiguous=out.flags.c_contiguous):
                before, parent_before = out.tobytes(), storage.tobytes()
                with self.assertRaises((ValueError, TypeError, BufferError)):
                    _sgd_accel.fill_rates(out, 0, 3, 8.0)
                self.assertEqual(out.tobytes(), before)
                self.assertEqual(storage.tobytes(), parent_before)
        with self.assertRaises((ValueError, TypeError, BufferError)):
            _sgd_accel.fill_rates([1., 2., 3.], 0, 3, 8.0)

    @requires_native
    def test_memoryview_buffer_is_supported_and_only_its_span_changes(self):
        storage = np.full(5, -17., dtype=np.float64)
        _sgd_accel.fill_rates(memoryview(storage[1:4]), 3, 3, 8.0)
        self.assertEqual(storage[1:4].tobytes(), original_rates(8.0, 3, 3).tobytes())
        self.assertEqual(storage[[0, 4]].tolist(), [-17., -17.])

    @requires_native
    def test_subnormal_and_overflow_fallback_match_all_numpy_error_policies(self):
        # Schedule arithmetic is Python float arithmetic, not a NumPy ufunc.
        # It must not acquire new NumPy warnings, callbacks or exceptions.
        cases = ((float(np.finfo(np.float64).max), LIMIT - 13, 13),
                 (1e-308, 0, 13), (float.fromhex('0x0.0000000000001p-1022'), 0, 13))

        class EventLog:
            def __init__(self):
                self.events = []

            def __call__(self, *args):
                self.events.append(args)

            def write(self, text):
                self.events.append(text)

        old_callback = np.geterrcall()
        try:
            for policy in ('ignore', 'warn', 'raise', 'call', 'print', 'log'):
                for B, updates, m in cases:
                    with self.subTest(policy=policy, B=B, updates=updates):
                        results = []
                        for candidate in (False, True):
                            events = EventLog()
                            np.seterrcall(events)
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always')
                                with np.errstate(all=policy):
                                    rates = (_native_optimizer._epoch_rates(_sgd_accel, B, updates, m)
                                             if candidate else original_rates(B, updates, m))
                            results.append((rates.tobytes(), events.events,
                                            [(item.category, str(item.message)) for item in caught]))
                        self.assertEqual(results[0], results[1])
        finally:
            np.seterrcall(old_callback)


class NativeRateFitTests(unittest.TestCase):
    def assert_same_fit(self, first, second):
        for name in ('initial_state_', 'final_state_', 'selected_state_', 'rng_state_',
                     'selected_iteration_', 'n_updates_', 'n_samples_seen_', 'learning_rate_B_',
                     'stop_reason_', 'model_status_', 'fit_usable_', 'cv_eligible_',
                     'kernel_evaluation_count_', 'kernel_cache_hits_', 'kernel_cache_misses_',
                     'kernel_cache_evictions_', 'kernel_cache_peak_bytes_', 'coef_', 'dual_coef_',
                     'intercept_', 'stage_seeds_', 'selected_indices_', 'selected_ids_',
                     'selected_class_counts_', 'selection_scores_'):
            if hasattr(first, name):
                self.assertEqual(pickle.dumps(getattr(first, name), protocol=5),
                                 pickle.dumps(getattr(second, name), protocol=5), name)
        if isinstance(first, TwoStageDWDClassifier):
            self.assert_same_fit(first.linear_estimator_, second.linear_estimator_)
            self.assert_same_fit(first.kernel_estimator_, second.kernel_estimator_)

    @requires_native
    def test_complete_fits_and_external_rng_match_original_rate_preparation(self):
        def old_rates(native, B, updates, m):
            return original_rates(B, updates, m)

        for m in (13, 14):
            for sampling in ('with_replacement', 'shuffle'):
                for constructor in (LinearSGDDWD, KernelSGDDWD, TwoStageDWDClassifier):
                    with self.subTest(m=m, sampling=sampling, model=constructor.__name__):
                        X, y = problem(m)
                        rng = np.random.Generator(np.random.Philox(2017))
                        rng.integers(3)
                        external = pickle.dumps(rng, protocol=5)
                        options = dict(max_epochs=5, random_state=rng, sampling=sampling,
                                       optimizer_backend='native')
                        if constructor is not LinearSGDDWD:
                            options['kernel_cache_bytes'] = m * m * 8
                        if constructor is TwoStageDWDClassifier:
                            options['retention_fraction'] = .8
                        with patch.object(_native_optimizer, '_epoch_rates', side_effect=old_rates):
                            original = constructor(**options).fit(X, y)
                        candidate = constructor(**options).fit(X, y)
                        self.assert_same_fit(original, candidate)
                        self.assertEqual(original.decision_function(X).tobytes(),
                                         candidate.decision_function(X).tobytes())
                        self.assertEqual(original.predict(X).tobytes(), candidate.predict(X).tobytes())
                        self.assertEqual(pickle.dumps(rng, protocol=5), external)

    @requires_native
    def test_fitted_rate_trace_uses_helper_for_both_stages(self):
        X, y = problem(14)
        fill = _sgd_accel.fill_rates
        requests = []

        def trace(out, updates, m, B):
            answer = fill(out, updates, m, B)
            self.assertEqual(out.tobytes(), original_rates(B, updates, m).tobytes())
            requests.append((updates, m, B))
            return answer

        with patch.object(_sgd_accel, 'fill_rates', side_effect=trace):
            model = TwoStageDWDClassifier(max_epochs=3, retention_fraction=.8,
                optimizer_backend='native', random_state=2017).fit(X, y)
        self.assertEqual(len(requests), 6)
        self.assertEqual([r[1] for r in requests[:3]], [14] * 3)
        self.assertEqual([r[1] for r in requests[3:]], [len(model.selected_indices_)] * 3)
        for segment in (requests[:3], requests[3:]):
            self.assertEqual([r[0] for r in segment], [i * segment[0][1] for i in range(3)])

    @requires_native
    def test_failed_update_prefix_and_rng_match_original_rates(self):
        original_column = _estimators._kernel_values
        X, y = problem(13)
        for sampling in ('with_replacement', 'shuffle'):
            for failed_call in (1, 6, 13, 29):
                with self.subTest(sampling=sampling, failed_call=failed_call):
                    rng = np.random.Generator(np.random.PCG64DXSM(191))
                    external = pickle.dumps(rng, protocol=5)
                    models = []
                    for old in (True, False):
                        calls = [0]

                        def fail_column(*args):
                            calls[0] += 1
                            if calls[0] == failed_call:
                                raise FloatingPointError('Controlled rate-prefix failure')
                            return original_column(*args)

                        options = dict(max_epochs=4, random_state=rng, sampling=sampling,
                                       optimizer_backend='native', kernel_cache_bytes=0)
                        model = KernelSGDDWD(**options)
                        rate_function = (_native_optimizer._epoch_rates if not old else
                            lambda native, B, updates, m: original_rates(B, updates, m))
                        with patch.object(_estimators, '_kernel_values', side_effect=fail_column), \
                                patch.object(_native_optimizer, '_epoch_rates', side_effect=rate_function):
                            with self.assertRaises(NumericalFailure):
                                model.fit(X, y)
                        self.assertEqual(model.n_updates_, failed_call - 1)
                        self.assertFalse(model.fit_usable_)
                        self.assertEqual(pickle.dumps(rng, protocol=5), external)
                        models.append(model)
                    self.assert_same_fit(*models)

    def test_rate_preparation_follows_sampling_and_does_not_retry_or_run_solver_on_error(self):
        X, y = problem(13)
        for sampling in ('with_replacement', 'shuffle'):
            for error in (ValueError('controlled rate shape failure'),
                          FloatingPointError('controlled rate arithmetic failure'),
                          MemoryError('controlled rate allocation failure'),
                          KeyboardInterrupt('controlled rate interruption')):
                with self.subTest(sampling=sampling, error=type(error).__name__):
                    rng = np.random.default_rng(73)
                    expected_rng = copy.deepcopy(rng)
                    if sampling == 'shuffle':
                        expected_rng.permutation(len(y))
                    else:
                        _native_optimizer._replacement_indices(expected_rng, len(y))

                    def fail_rates(native, B, updates, m):
                        self.assertEqual((updates, m), (0, len(y)))
                        self.assertEqual(pickle.dumps(rng.bit_generator.state),
                                         pickle.dumps(expected_rng.bit_generator.state))
                        raise error

                    native = SimpleNamespace(run_steps=Mock())
                    est = SimpleNamespace(max_epochs=1, sampling=sampling)
                    with patch.object(_native_optimizer, '_epoch_rates', side_effect=fail_rates) as rates:
                        with self.assertRaises(type(error)) as caught:
                            _native_optimizer.optimize('linear', X, y, rng, est, native)
                    self.assertIs(caught.exception, error)
                    rates.assert_called_once()
                    native.run_steps.assert_not_called()

    def test_zero_epochs_do_not_prepare_rates_sample_or_call_native_steps(self):
        X, y = problem(13)
        for sampling in ('with_replacement', 'shuffle'):
            with self.subTest(sampling=sampling):
                rng = np.random.default_rng(73)
                original_rng = pickle.dumps(rng.bit_generator.state)
                native = SimpleNamespace(run_steps=Mock())
                est = SimpleNamespace(max_epochs=0, sampling=sampling)
                with patch.object(_native_optimizer, '_epoch_rates',
                                  side_effect=AssertionError('No rates for zero epochs')):
                    result = _native_optimizer.optimize('linear', X, y, rng, est, native)
                self.assertEqual(result['n_updates_'], 0)
                self.assertEqual(pickle.dumps(rng.bit_generator.state), original_rng)
                native.run_steps.assert_not_called()

    def test_zero_epochs_still_validate_the_learning_rate_bound(self):
        X, y = problem(13)
        for lambd, diagonal in ((float('inf'), 1.0),
                                (float(np.finfo(np.float64).max), 1.0),
                                (1e-6, float('inf'))):
            with self.subTest(lambd=lambd, diagonal=diagonal):
                rng = np.random.default_rng(73)
                original_rng = pickle.dumps(rng.bit_generator.state)
                native = SimpleNamespace(run_steps=Mock())
                est = SimpleNamespace(max_epochs=0, sampling='with_replacement',
                                      lambd_=lambd, max_kernel_diagonal_=diagonal)
                with patch.object(_native_optimizer, '_epoch_rates') as rates:
                    with self.assertRaisesRegex(NumericalFailure, 'learning-rate bound'):
                        _native_optimizer.optimize('kernel', X, y, rng, est, native)
                self.assertEqual(pickle.dumps(rng.bit_generator.state), original_rng)
                rates.assert_not_called()
                native.run_steps.assert_not_called()


if __name__ == '__main__':
    unittest.main()
