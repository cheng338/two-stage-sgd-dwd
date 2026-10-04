"""Exact kernel updates and failure atomicity when finite checks share a pass."""
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from two_stage_sgd_dwd import KernelSGDDWD, NumericalFailure, TwoStageDWDClassifier
from two_stage_sgd_dwd import _estimators

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None

requires_native = unittest.skipIf(_sgd_accel is None, 'Optional compiled extension unavailable')


def oracle_step(weights, bias, column, index, label, rate, lambd):
    """Independent scalar formula, preserving separate specified operations."""
    margin = float(label) * (float(np.dot(weights, column)) + bias)
    if not np.isfinite(margin):
        raise FloatingPointError('Nonfinite margin')
    derivative = (-.25 / margin) / margin if margin > .5 else -1.
    q = derivative * float(label)
    multiplier = float(rate) * q
    next_bias = bias - multiplier
    shrink = 1. - float(rate) * (2. * lambd)
    if not all(np.isfinite(x) for x in (multiplier, next_bias, shrink)):
        raise FloatingPointError('Nonfinite scalar update')
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        proposed = shrink * weights
        proposed[index] -= multiplier
    if not np.all(np.isfinite(proposed)):
        raise FloatingPointError('Nonfinite coefficients')
    return proposed, next_bias


def oracle_prefix(initial, bias, rows, labels, indices, rates, lambd):
    weights = np.array(initial, dtype=np.float64, copy=True)
    for index, rate in zip(indices, rates):
        weights, bias = oracle_step(weights, bias, rows[int(index)], int(index),
                                    labels[index], rate, lambd)
    return weights, bias


class CapturedScratch(np.ndarray):
    """Expose a custom proposal buffer to verify the legacy failure path."""
    proposal = None

    def __array_function__(self, function, types, args, kwargs):
        if function is np.copy:
            return np.array(np.asarray(args[0]), copy=True).view(type(self))
        if function is np.empty_like:
            result = np.empty_like(np.asarray(args[0])).view(type(self))
            type(self).proposal = result
            return result
        if function is np.dot:
            return np.dot(np.asarray(args[0]), np.asarray(args[1]))
        return super().__array_function__(function, types, args, kwargs)


class MutatingRow(np.ndarray):
    """A dot hook can invalidate current coefficients before the shrink pass."""
    def __array_function__(self, function, types, args, kwargs):
        if function is np.dot:
            args[0][1] = float('inf')
            return np.float64(0.)
        return super().__array_function__(function, types, args, kwargs)


@requires_native
class NativeKernelFinitePassTests(unittest.TestCase):
    def exact(self, actual, expected):
        first, second = np.asarray(actual), np.asarray(expected)
        self.assertEqual(first.dtype, second.dtype)
        self.assertEqual(first.shape, second.shape)
        self.assertEqual(first.tobytes(), second.tobytes())

    def every_prefix(self, initial, bias, rows, labels, indices, rates, lambd):
        inputs = (initial, rows, labels, indices, rates)
        originals = [value.tobytes() for value in inputs]
        for stop in range(len(indices) + 1):
            with self.subTest(prefix=stop):
                calls = []

                def get_row(index):
                    calls.append(index)
                    return rows[index]

                actual, result_bias, completed, error = _sgd_accel.run_steps(
                    1, initial, bias, get_row, labels, indices[:stop], rates[:stop], lambd)
                expected, expected_bias = oracle_prefix(
                    initial, bias, rows, labels, indices[:stop], rates[:stop], lambd)
                self.exact(actual, expected)
                self.exact(np.float64(result_bias), np.float64(expected_bias))
                self.assertEqual(completed, stop)
                self.assertIsNone(error)
                self.assertEqual(calls, indices[:stop].tolist())
                self.assertFalse(np.shares_memory(actual, initial))
                self.assertEqual([value.tobytes() for value in inputs], originals)

    def test_shrink_sign_zero_subnormals_and_odd_widths_match_every_prefix(self):
        tiny = float(np.nextafter(0., 1.))
        cases = [
            ('signed_zero', np.array([0., -0., 0.]), np.array([0., .25, .5, .75]), 1.),
            ('subnormal_products', np.array([tiny, -tiny, 3 * tiny]),
             np.array([0., .25, .5, .75]), 1.),
            ('negative_shrink', np.array([.4, -.7, .2]), np.array([.7, .4, .1, 0.]), 2.),
            ('small_width', np.array([.7]), np.array([0., .25, .5, .75]), 1.),
        ]
        for name, initial, rates, lambd in cases:
            with self.subTest(case=name):
                n = len(initial)
                rows = np.zeros((n, n))
                labels = np.where(np.arange(n) % 2, -1., 1.)
                indices = np.array([j % n for j in range(len(rates))], dtype=np.int64)
                self.every_prefix(initial, -.2, rows, labels, indices, rates, lambd)

    def test_loss_boundary_tail_and_nontrivial_columns_match_every_prefix(self):
        initial = np.array([.3, -.2, .1, -.4, .2])
        rows = np.array([[1., .4, .2, .1, .3], [.4, 1., .1, .2, .4],
                         [.2, .1, 1., .3, .2], [.1, .2, .3, 1., .1],
                         [.3, .4, .2, .1, 1.]])
        labels = np.array([-1., 1., 1., -1., 1.])
        indices = np.array([4, 1, 3, 0, 2, 4], dtype=np.int64)
        rates = np.array([.12, .04, .3, .7, 0., .2])
        self.every_prefix(initial, .17, rows, labels, indices, rates, .31)
        for margin in (np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.), 8., 1e150):
            for label in (-1., 1.):
                with self.subTest(margin=margin, label=label):
                    self.every_prefix(np.array([0.]), margin / label, np.zeros((1, 1)),
                        np.array([label]), np.array([0], dtype=np.int64), np.array([.13]), .07)

    def test_target_and_other_shrink_overflow_do_not_commit_failed_update(self):
        maximum = float(np.finfo(np.float64).max)
        for large_index in (0, 2):
            for prefix in (0, 2, 4):
                with self.subTest(large_index=large_index, prefix=prefix):
                    initial = np.array([.1, -.2, .3])
                    initial[large_index] = maximum
                    rows = np.zeros((3, 3))
                    labels = np.ones(3)
                    indices = np.zeros(prefix + 1, dtype=np.int64)
                    rates = np.array([.01] * prefix + [2.])
                    before = initial.tobytes()
                    expected, expected_bias = oracle_prefix(
                        initial, 0., rows, labels, indices[:prefix], rates[:prefix], 1.)
                    with self.assertRaises(FloatingPointError):
                        oracle_step(expected, expected_bias, rows[0], 0, 1., 2., 1.)
                    actual, bias, completed, error = _sgd_accel.run_steps(
                        1, initial, 0., rows.__getitem__, labels, indices, rates, 1.)
                    self.exact(actual, expected)
                    self.exact(np.float64(bias), np.float64(expected_bias))
                    self.assertEqual(completed, prefix)
                    self.assertEqual(error, 'Nonfinite coefficients')
                    self.assertEqual(initial.tobytes(), before)

    def test_finite_shrink_product_then_indexed_subtraction_overflow_is_detected(self):
        maximum = float(np.finfo(np.float64).max)
        lambd = float(np.nextafter(0., 1.))
        rate = .75 * maximum
        initial = np.array([maximum, .2])
        rows = np.zeros((2, 2))
        labels = np.ones(2)
        self.assertTrue(np.all(np.isfinite((1. - rate * (2. * lambd)) * initial)))
        with self.assertRaises(FloatingPointError):
            oracle_step(initial, 0., rows[0], 0, 1., rate, lambd)
        actual, bias, completed, error = _sgd_accel.run_steps(
            1, initial, 0., rows.__getitem__, labels, np.array([0], dtype=np.int64),
            np.array([rate]), lambd)
        self.exact(actual, initial)
        self.exact(np.float64(bias), np.float64(0.))
        self.assertEqual(completed, 0)
        self.assertEqual(error, 'Nonfinite coefficients')

    def test_initial_and_scalar_errors_keep_existing_failure_categories(self):
        for value in (float('inf'), -float('inf'), float('nan')):
            with self.subTest(initial=value):
                initial = np.array([value])
                before = initial.tobytes()
                with self.assertRaisesRegex(ValueError, 'Nonfinite initial coefficients'):
                    _sgd_accel.run_steps(1, initial, 0., lambda index: np.zeros(1),
                        np.ones(1), np.array([0], dtype=np.int64), np.array([.1]), 1.)
                self.assertEqual(initial.tobytes(), before)
        for initial, column, bias, rate, lambd, expected_error in (
                (np.zeros(1), np.zeros(1), 0., 1., float(np.finfo(np.float64).max),
                 'Nonfinite shrinkage'),
                (np.ones(1), np.array([-1e308]), 1e308, 1e308, 1.,
                 'Nonfinite scalar update')):
            with self.subTest(error=expected_error):
                actual, result_bias, completed, error = _sgd_accel.run_steps(
                    1, initial, bias, lambda index: column, np.ones(1),
                    np.array([0], dtype=np.int64), np.array([rate]), lambd)
                self.exact(actual, initial)
                self.exact(np.float64(result_bias), np.float64(bias))
                self.assertEqual(completed, 0)
                self.assertEqual(error, expected_error)

    def test_custom_scratch_keeps_full_product_writes_and_subtraction_before_rejection(self):
        initial = np.array([np.finfo(np.float64).max, .2, .3]).view(CapturedScratch)
        before = initial.tobytes()
        CapturedScratch.proposal = None
        try:
            actual, bias, completed, error = _sgd_accel.run_steps(
                1, initial, 0., lambda index: np.zeros(3), np.ones(3),
                np.array([2], dtype=np.int64), np.array([2.]), 1.)
            with np.errstate(over='ignore', invalid='ignore'):
                expected_proposal = -3. * np.asarray(initial)
                expected_proposal[2] -= -2.
            self.assertIsInstance(actual, CapturedScratch)
            self.exact(CapturedScratch.proposal, expected_proposal)
            self.exact(actual, np.asarray(initial))
            self.assertEqual(completed, 0)
            self.assertEqual(bias, 0.)
            self.assertEqual(error, 'Nonfinite coefficients')
            self.assertEqual(initial.tobytes(), before)
        finally:
            CapturedScratch.proposal = None

    def test_custom_array_successful_updates_still_match_oracle(self):
        initial = np.array([.2, -.1, .4]).view(CapturedScratch)
        rows = np.array([[1., .2, .1], [.2, 1., .3], [.1, .3, 1.]])
        try:
            self.every_prefix(initial, .1, rows, np.array([-1., 1., 1.]),
                np.array([0, 2, 1], dtype=np.int64), np.array([.1, .7, .2]), 1.)
        finally:
            CapturedScratch.proposal = None

    def test_custom_row_dot_mutation_cannot_bypass_product_finiteness_check(self):
        initial = np.array([.1, .2])
        before = initial.tobytes()
        row = np.zeros(2).view(MutatingRow)
        actual, bias, completed, error = _sgd_accel.run_steps(
            1, initial, 0., lambda index: row, np.ones(2),
            np.array([0], dtype=np.int64), np.array([.1]), 1.)
        # The hook intentionally mutates the private current copy, as it did
        # before this optimization. The update must nevertheless remain failed.
        self.assertTrue(np.isinf(actual[1]))
        self.assertEqual(actual[0], initial[0])
        self.assertEqual(completed, 0)
        self.assertEqual(bias, 0.)
        self.assertEqual(error, 'Nonfinite coefficients')
        self.assertEqual(initial.tobytes(), before)

    def test_loader_errors_preserve_prefix_and_structural_errors_propagate(self):
        rows = np.array([[1., .2], [.2, 1.]])
        initial = np.array([.2, -.1])
        labels = np.array([-1., 1.])
        indices = np.array([0, 1, 1, 0], dtype=np.int64)
        rates = np.array([.1, .2, .3, .4])
        for prefix in (0, 2, 3):
            with self.subTest(prefix=prefix):
                calls = []

                def get_row(index):
                    calls.append(index)
                    if len(calls) == prefix + 1:
                        raise FloatingPointError('Controlled kernel column failure')
                    return rows[index]

                actual, bias, completed, error = _sgd_accel.run_steps(
                    1, initial, .1, get_row, labels, indices, rates, .2)
                expected, expected_bias = oracle_prefix(
                    initial, .1, rows, labels, indices[:prefix], rates[:prefix], .2)
                self.exact(actual, expected)
                self.exact(np.float64(bias), np.float64(expected_bias))
                self.assertEqual(completed, prefix)
                self.assertEqual(calls, indices[:prefix + 1].tolist())
                self.assertEqual(error, 'Controlled kernel column failure')
        with self.assertRaisesRegex(KeyError, 'Structural column error'):
            _sgd_accel.run_steps(1, initial, .1, MockStructuralLoader(), labels, indices, rates, .2)

    def assert_same_state(self, first, second):
        for name in ('initial_state_', 'final_state_', 'selected_state_', 'rng_state_',
                     'n_updates_', 'n_samples_seen_', 'learning_rate_B_', 'model_status_',
                     'stop_reason_', 'fit_usable_', 'cv_eligible_', 'kernel_evaluation_count_',
                     'kernel_cache_hits_', 'kernel_cache_misses_', 'kernel_cache_evictions_',
                     'kernel_cache_peak_bytes_', 'kernel_cache_capacity_columns_'):
            if hasattr(first, name):
                self.assertEqual(pickle.dumps(getattr(first, name), protocol=5),
                                 pickle.dumps(getattr(second, name), protocol=5), name)

    def test_public_fits_sampling_cache_and_two_stage_state_match_python(self):
        x = np.linspace(-1.5, 1.5, 14)
        X = np.column_stack((x, .13 * np.cos(np.arange(14)), x * x / 7))
        y = np.where(x < 0, -1, 1)
        for constructor in (KernelSGDDWD, TwoStageDWDClassifier):
            for sampling in ('with_replacement', 'shuffle'):
                for cache in (0, 2 * len(y) * 8, len(y) ** 2 * 8):
                    with self.subTest(model=constructor.__name__, sampling=sampling, cache=cache):
                        rng = np.random.Generator(np.random.Philox(2017))
                        external = pickle.dumps(rng, protocol=5)
                        settings = dict(max_epochs=3, sampling=sampling, random_state=rng,
                                        kernel_cache_bytes=cache)
                        if constructor is TwoStageDWDClassifier:
                            settings['retention_fraction'] = .8
                        reference = constructor(**settings).fit(X, y)
                        native = constructor(optimizer_backend='native', **settings).fit(X, y)
                        stages = [(reference, native)]
                        if constructor is TwoStageDWDClassifier:
                            stages += [(reference.linear_estimator_, native.linear_estimator_),
                                       (reference.kernel_estimator_, native.kernel_estimator_)]
                            for name in ('stage_seeds_', 'selected_indices_', 'selected_ids_',
                                         'selected_class_counts_', 'selection_scores_'):
                                self.assertEqual(pickle.dumps(getattr(reference, name), protocol=5),
                                                 pickle.dumps(getattr(native, name), protocol=5), name)
                        for first, second in stages:
                            self.assert_same_state(first, second)
                        self.exact(reference.decision_function(X), native.decision_function(X))
                        self.exact(reference.predict(X), native.predict(X))
                        self.assertEqual(pickle.dumps(rng, protocol=5), external)

    def test_public_later_epoch_failure_keeps_rng_state_and_evaluation_counts(self):
        X = np.column_stack((np.linspace(-1., 1., 8), np.cos(np.arange(8))))
        y = np.array([-1] * 4 + [1] * 4)
        original = _estimators._kernel_values
        for sampling in ('with_replacement', 'shuffle'):
            for failed_call in (1, len(y), len(y) + 3):
                with self.subTest(sampling=sampling, failed_call=failed_call):
                    models = []
                    rng = np.random.Generator(np.random.PCG64DXSM(42))
                    external = pickle.dumps(rng, protocol=5)
                    for backend in ('python', 'native'):
                        calls = [0]

                        def failing_column(*args):
                            calls[0] += 1
                            if calls[0] == failed_call:
                                raise FloatingPointError('Controlled column failure')
                            return original(*args)

                        model = KernelSGDDWD(max_epochs=3, random_state=rng,
                                             sampling=sampling, optimizer_backend=backend)
                        with patch.object(_estimators, '_kernel_values', side_effect=failing_column):
                            with self.assertRaises(NumericalFailure):
                                model.fit(X, y)
                        self.assertEqual(model.n_updates_, failed_call - 1)
                        self.assertEqual(model.kernel_evaluation_count_, (failed_call - 1) * len(y))
                        self.assertFalse(model.fit_usable_)
                        self.assertFalse(model.cv_eligible_)
                        self.assertEqual(pickle.dumps(rng, protocol=5), external)
                        models.append(model)
                    self.assert_same_state(*models)


class MockStructuralLoader:
    def __call__(self, index):
        raise KeyError('Structural column error')


if __name__ == '__main__':
    unittest.main()
