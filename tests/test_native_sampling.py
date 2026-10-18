"""Exact replacement streams, guarded dispatch, and native failed-prefix parity.

These are implementation checks for the explicitly supported NumPy versions,
not a claim that NumPy guarantees scalar/array stream equivalence in general.
"""
import copy
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from two_stage_sgd_dwd import (
    KernelSGDDWD, LinearSGDDWD, NumericalFailure, TwoStageDWDClassifier,
)
from two_stage_sgd_dwd import _estimators, _native_optimizer

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None

BITS = (np.random.PCG64, np.random.PCG64DXSM, np.random.Philox,
        np.random.SFC64, np.random.MT19937)
SUPPORTED_NUMPY_VERSIONS = ('2.5.2', '2.5.3', '2.5.4')
requires_native = unittest.skipIf(_sgd_accel is None, "Optional compiled extension unavailable")


def state_bytes(rng):
    return pickle.dumps(rng.bit_generator.state, protocol=5)


def advanced_generator(bit_type, seed, advance):
    rng = np.random.Generator(bit_type(seed))
    if advance:
        # Exercise partially used 32-bit caches, Philox blocks, and MT positions.
        rng.integers(17, size=advance)
        rng.random(3)
        rng.bit_generator.random_raw(advance % 5)
        rng.integers(3)
    return rng


def scalar_indices(rng, m):
    return np.asarray([int(rng.integers(m)) for _ in range(m)], dtype=np.int64)


def problem(m):
    x = np.linspace(-1.5, 1.5, m)
    X = np.column_stack((x, .13 * np.cos(np.arange(m)), x * x / 7))
    return X, np.where(x < 0, -1, 1)


class AlteredGenerator(np.random.Generator):
    """An override whose sized and scalar calls intentionally disagree."""
    def integers(self, high, size=None, **kwargs):
        self.requests.append(size)
        if size is not None:
            return np.full(size, -999, dtype=np.int64)
        return int(super().integers(high, **kwargs)) + 1000


class DerivedPCG64(np.random.PCG64):
    pass


class ReplacementStreamTests(unittest.TestCase):
    def assert_stream(self, actual, expected, rng, oracle):
        self.assertEqual(actual.dtype, np.dtype(np.int64))
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual(state_bytes(rng), state_bytes(oracle))
        # Continuation must include both the bounded generator and raw state.
        self.assertEqual(rng.integers(2**32 + 1, size=9).tobytes(),
                         oracle.integers(2**32 + 1, size=9).tobytes())
        self.assertEqual(rng.bit_generator.random_raw(3).tobytes(),
                         oracle.bit_generator.random_raw(3).tobytes())
        self.assertEqual(state_bytes(rng), state_bytes(oracle))

    def test_helper_preserves_output_complete_state_and_continuation(self):
        for bit_type in BITS:
            for seed in (0, 42, 20261004):
                for m in (1, 2, 3, 7, 8, 17, 31, 32, 33, 255, 256, 257, 725, 2417):
                    for advance in (0, 1, 2, 623, 624):
                        with self.subTest(bit=bit_type.__name__, seed=seed, m=m, advance=advance):
                            rng = advanced_generator(bit_type, seed, advance)
                            oracle = copy.deepcopy(rng)
                            for _ in range(3):
                                actual = _native_optimizer._replacement_indices(rng, m)
                                expected = scalar_indices(oracle, m)
                                self.assert_stream(actual, expected, rng, oracle)

    @unittest.skipUnless(np.__version__ in SUPPORTED_NUMPY_VERSIONS,
                         'Scalar/array evidence requires an actual supported NumPy release')
    def test_wide_bounds_use_small_independent_draw_counts(self):
        # The helper's bound equals its length. Probe huge bounds directly so
        # these dependency checks never attempt enormous helper allocations.
        # Powers of two and their neighbors include high-rejection intervals,
        # the uint32/uint64 dispatch boundary, and the largest int64 range.
        bounds = (1, 2, 3, 7, 8, 9, 15, 16, 17, 255, 256, 257, 725, 2417,
                  10000, 60000, 2**16 - 1, 2**16, 2**16 + 1,
                  2**31 - 1, 2**31, 2**31 + 1,
                  2**32 - 1, 2**32, 2**32 + 1,
                  2**62 - 1, 2**62, 2**62 + 1, 2**63 - 1, 2**63)
        for bit_type in BITS:
            for bound in bounds:
                for count in (1, 2, 3, 17, 64, 65, 257):
                    for advance in (0, 1, 2, 623, 624):
                        with self.subTest(bit=bit_type.__name__, bound=bound,
                                          count=count, advance=advance):
                            rng = advanced_generator(bit_type, 20261004, advance)
                            oracle = copy.deepcopy(rng)
                            actual = rng.integers(bound, size=count, dtype=np.int64)
                            expected = np.asarray([int(oracle.integers(bound))
                                                   for _ in range(count)], dtype=np.int64)
                            self.assert_stream(actual, expected, rng, oracle)

    @unittest.skipUnless(np.__version__ in SUPPORTED_NUMPY_VERSIONS,
                         'Fast-path dispatch is version-specific')
    def test_known_builtin_generators_do_not_take_the_scalar_list_path(self):
        for bit_type in BITS:
            with self.subTest(bit=bit_type.__name__):
                rng = advanced_generator(bit_type, 9, 2)
                oracle = copy.deepcopy(rng)
                original_asarray = np.asarray
                with patch.object(_native_optimizer.np, 'asarray', wraps=original_asarray) as array:
                    result = _native_optimizer._replacement_indices(rng, 11)
                list_calls = [call for call in array.call_args_list
                              if isinstance(call.args[0], list)]
                self.assertEqual(list_calls, [])
                self.assert_stream(result, scalar_indices(oracle, 11), rng, oracle)

    def test_other_versions_preserve_the_original_scalar_fallback(self):
        # Version spoofing checks only exclusion from dispatch. The positive
        # stream evidence above runs against the installed NumPy implementation.
        for version in ('2.5.1', '2.5.5', '2.6.0', '2.5.2+custom',
                        '2.5.3+custom', '2.5.3rc1', '2.5.4+custom',
                        '2.5.4rc1', '2.5.4.dev0', 'unknown'):
            for bit_type in BITS:
                with self.subTest(version=version, bit=bit_type.__name__):
                    rng = advanced_generator(bit_type, 27, 1)
                    oracle = copy.deepcopy(rng)
                    original_asarray = np.asarray
                    with patch.object(_native_optimizer.np, '__version__', version), \
                            patch.object(_native_optimizer.np, 'asarray', wraps=original_asarray) as array:
                        actual = _native_optimizer._replacement_indices(rng, 13)
                    list_calls = [call for call in array.call_args_list
                                  if isinstance(call.args[0], list)]
                    self.assertEqual(len(list_calls), 1)
                    self.assert_stream(actual, scalar_indices(oracle, 13), rng, oracle)

    def test_generator_subclass_override_receives_only_scalar_calls(self):
        for bit_type in BITS:
            with self.subTest(bit=bit_type.__name__):
                rng = AlteredGenerator(bit_type(5))
                rng.requests = []
                oracle = AlteredGenerator(bit_type(5))
                oracle.requests = []
                actual = _native_optimizer._replacement_indices(rng, 13)
                expected = scalar_indices(oracle, 13)
                self.assertEqual(rng.requests, [None] * 13)
                self.assertEqual(actual.tobytes(), expected.tobytes())
                self.assertEqual(state_bytes(rng), state_bytes(oracle))
                self.assertTrue(np.all(actual >= 1000))

    def test_bit_generator_subclass_is_not_admitted_by_base_class_membership(self):
        rng = np.random.Generator(DerivedPCG64(7))
        # NumPy 1 pickles custom bit generators by registry name. This helper
        # never copies the generator, so construct its oracle independently.
        oracle = np.random.Generator(DerivedPCG64(7))
        self.assertIs(type(oracle), type(rng))
        self.assertIs(type(oracle.bit_generator), type(rng.bit_generator))
        self.assertEqual(state_bytes(oracle), state_bytes(rng))
        original_asarray = np.asarray
        with patch.object(_native_optimizer.np, 'asarray', wraps=original_asarray) as array:
            actual = _native_optimizer._replacement_indices(rng, 17)
        list_calls = [call for call in array.call_args_list if isinstance(call.args[0], list)]
        self.assertEqual(len(list_calls), 1)
        self.assert_stream(actual, scalar_indices(oracle, 17), rng, oracle)

    def test_custom_generator_protocol_preserves_scalar_override(self):
        class ScalarOnly:
            def __init__(self):
                self.calls = []

            def integers(self, high, *args, **kwargs):
                self.calls.append((high, args, kwargs))
                if args or kwargs:
                    raise AssertionError('Custom generator must not receive a sized request')
                return len(self.calls) % high

        rng = ScalarOnly()
        actual = _native_optimizer._replacement_indices(rng, 5)
        self.assertEqual(actual.tolist(), [1, 2, 3, 4, 0])
        self.assertEqual(rng.calls, [(5, (), {})] * 5)


class NativeSamplingIntegrationTests(unittest.TestCase):
    def assert_same_fit(self, a, b):
        for name in ('initial_state_', 'final_state_', 'selected_state_', 'rng_state_',
                     'selected_iteration_', 'n_updates_', 'n_samples_seen_',
                     'learning_rate_B_', 'stop_reason_', 'model_status_',
                     'cv_eligible_', 'fit_usable_', 'termination_reason_',
                     'kernel_evaluation_count_', 'kernel_cache_hits_',
                     'kernel_cache_misses_', 'kernel_cache_evictions_',
                     'kernel_cache_peak_bytes_', 'coef_', 'dual_coef_', 'intercept_'):
            if hasattr(a, name):
                self.assertEqual(pickle.dumps(getattr(a, name), protocol=5),
                                 pickle.dumps(getattr(b, name), protocol=5), name)

    @requires_native
    def test_epoch_traces_match_scalar_oracle_for_every_builtin_generator(self):
        actual_steps = _sgd_accel.run_steps
        for bit_type in BITS:
            for m in (13, 14):
                for sampling in ('with_replacement', 'shuffle'):
                    with self.subTest(bit=bit_type.__name__, m=m, sampling=sampling):
                        X, y = problem(m)
                        rng = advanced_generator(bit_type, 31, 3)
                        external = pickle.dumps(rng, protocol=5)
                        oracle = copy.deepcopy(rng)
                        requests = []

                        def trace(kind, coefficients, bias, get_row, labels, indices, rates, lambd):
                            requests.append((indices.copy(), rates.copy()))
                            return actual_steps(kind, coefficients, bias, get_row,
                                                labels, indices, rates, lambd)

                        with patch.object(_sgd_accel, 'run_steps', side_effect=trace):
                            model = KernelSGDDWD(max_epochs=7, random_state=rng, sampling=sampling,
                                                 optimizer_backend='native').fit(X, y)
                        self.assertEqual(len(requests), 7)
                        for epoch, (indices, rates) in enumerate(requests):
                            expected = (np.asarray(oracle.permutation(m), dtype=np.int64)
                                        if sampling == 'shuffle' else scalar_indices(oracle, m))
                            expected_rates = np.asarray([(1.0 / model.learning_rate_B_) *
                                (1.0 + (epoch * m + j) / m) ** (-.75) for j in range(m)])
                            self.assertEqual(indices.tobytes(), expected.tobytes())
                            self.assertEqual(rates.tobytes(), expected_rates.tobytes())
                        self.assertEqual(pickle.dumps(model.rng_state_, protocol=5), state_bytes(oracle))
                        self.assertEqual(pickle.dumps(rng, protocol=5), external)

    @requires_native
    def test_public_fit_parity_across_generators_sampling_and_cache_capacities(self):
        for bit_type in BITS:
            for sampling in ('with_replacement', 'shuffle'):
                for constructor in (LinearSGDDWD, KernelSGDDWD, TwoStageDWDClassifier):
                    capacities = (None,) if constructor is LinearSGDDWD else (0, 13 * 8 * 3, 13 * 13 * 8)
                    for capacity in capacities:
                        with self.subTest(bit=bit_type.__name__, sampling=sampling,
                                          model=constructor.__name__, capacity=capacity):
                            X, y = problem(13)
                            rng = advanced_generator(bit_type, 2017, 1)
                            external = pickle.dumps(rng, protocol=5)
                            options = dict(max_epochs=5, random_state=rng, sampling=sampling)
                            if capacity is not None:
                                options['kernel_cache_bytes'] = capacity
                            if constructor is TwoStageDWDClassifier:
                                options['retention_fraction'] = .8
                            baseline = constructor(**options).fit(X, y)
                            native = constructor(optimizer_backend='native', **options).fit(X, y)
                            pairs = [(baseline, native)]
                            if constructor is TwoStageDWDClassifier:
                                pairs = [(baseline.linear_estimator_, native.linear_estimator_),
                                         (baseline.kernel_estimator_, native.kernel_estimator_)]
                                for name in ('stage_seeds_', 'selected_indices_', 'selected_ids_',
                                             'selected_class_counts_', 'selection_scores_'):
                                    self.assertEqual(pickle.dumps(getattr(baseline, name), protocol=5),
                                                     pickle.dumps(getattr(native, name), protocol=5), name)
                            for first, second in pairs:
                                self.assert_same_fit(first, second)
                            self.assertEqual(baseline.decision_function(X).tobytes(),
                                             native.decision_function(X).tobytes())
                            self.assertEqual(baseline.predict(X).tobytes(), native.predict(X).tobytes())
                            self.assertEqual(pickle.dumps(rng, protocol=5), external)

    @requires_native
    def test_shuffle_never_calls_replacement_helper(self):
        X, y = problem(14)
        with patch.object(_native_optimizer, '_replacement_indices',
                          side_effect=AssertionError('Unexpected replacement sampling')):
            model = LinearSGDDWD(sampling='shuffle', max_epochs=3,
                                 optimizer_backend='native', random_state=7).fit(X, y)
        self.assertEqual(model.n_updates_, 42)

    @requires_native
    def test_numerical_failure_keeps_only_successful_updates_and_consumes_failed_draw(self):
        original = _estimators._kernel_values
        m = 13
        X, y = problem(m)
        for bit_type in BITS:
            for sampling in ('with_replacement', 'shuffle'):
                for failed_call in (1, 2, m // 2, m - 1, m, m + 1, 2 * m + 3):
                    with self.subTest(bit=bit_type.__name__, sampling=sampling, fail=failed_call):
                        rng = advanced_generator(bit_type, 191, 3)
                        external = pickle.dumps(rng, protocol=5)
                        oracle = copy.deepcopy(rng)
                        if sampling == 'shuffle':
                            for _ in range((failed_call - 1) // m + 1):
                                oracle.permutation(m)
                        else:
                            for _ in range(failed_call):
                                oracle.integers(m)
                        models = []
                        for backend in ('python', 'native'):
                            count = [0]

                            def fail_column(*args):
                                count[0] += 1
                                if count[0] == failed_call:
                                    raise FloatingPointError('Controlled sampling-prefix failure')
                                return original(*args)

                            model = KernelSGDDWD(max_epochs=4, random_state=rng, sampling=sampling,
                                                 optimizer_backend=backend, kernel_cache_bytes=0)
                            with patch.object(_estimators, '_kernel_values', side_effect=fail_column):
                                with self.assertRaises(NumericalFailure):
                                    model.fit(X, y)
                            self.assertEqual(model.n_updates_, failed_call - 1)
                            self.assertEqual(count[0], failed_call)
                            self.assertFalse(model.fit_usable_)
                            self.assertFalse(model.cv_eligible_)
                            self.assertEqual(pickle.dumps(model.rng_state_, protocol=5), state_bytes(oracle))
                            resumed = np.random.Generator(bit_type(0))
                            resumed.bit_generator.state = copy.deepcopy(model.rng_state_)
                            continued = copy.deepcopy(oracle)
                            self.assertEqual(resumed.integers(m, size=17).tobytes(),
                                             continued.integers(m, size=17).tobytes())
                            self.assertEqual(resumed.bit_generator.random_raw(5).tobytes(),
                                             continued.bit_generator.random_raw(5).tobytes())
                            self.assertEqual(state_bytes(resumed), state_bytes(continued))
                            self.assertEqual(pickle.dumps(rng, protocol=5), external)
                            models.append(model)
                        self.assert_same_fit(*models)

    @requires_native
    def test_cached_column_failure_preserves_prefix_and_external_rng(self):
        X, y = problem(13)
        original = _estimators._kernel_values
        for sampling in ('with_replacement', 'shuffle'):
            for capacity in (13 * 8 * 2, 13 * 13 * 8):
                with self.subTest(sampling=sampling, capacity=capacity):
                    rng = advanced_generator(np.random.Philox, 2017, 3)
                    external = pickle.dumps(rng, protocol=5)
                    models = []
                    for backend in ('python', 'native'):
                        calls = [0]

                        def fail_miss(*args):
                            calls[0] += 1
                            if calls[0] == 3:
                                raise FloatingPointError('Controlled cache-miss failure')
                            return original(*args)

                        model = KernelSGDDWD(max_epochs=4, random_state=rng, sampling=sampling,
                                             optimizer_backend=backend, kernel_cache_bytes=capacity)
                        with patch.object(_estimators, '_kernel_values', side_effect=fail_miss):
                            with self.assertRaises(NumericalFailure):
                                model.fit(X, y)
                        self.assertEqual(calls[0], 3)
                        self.assertFalse(model.fit_usable_)
                        self.assertEqual(pickle.dumps(rng, protocol=5), external)
                        models.append(model)
                    self.assert_same_fit(*models)

    @requires_native
    def test_fallback_version_preserves_native_fit(self):
        X, y = problem(14)
        expected = KernelSGDDWD(max_epochs=3, random_state=28).fit(X, y)
        with patch.object(_native_optimizer.np, '__version__', 'unvalidated-version'):
            actual = KernelSGDDWD(max_epochs=3, random_state=28,
                                  optimizer_backend='native').fit(X, y)
        self.assert_same_fit(expected, actual)
        self.assertEqual(expected.decision_function(X).tobytes(), actual.decision_function(X).tobytes())


if __name__ == '__main__':
    unittest.main()
