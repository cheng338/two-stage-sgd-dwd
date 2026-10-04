"""Exact native RBF columns, routing, counters, and failed-prefix behavior."""
import copy
from contextlib import contextmanager, ExitStack
import pickle
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn import config_context

from two_stage_sgd_dwd import KernelSGDDWD, NumericalFailure, TwoStageDWDClassifier
from two_stage_sgd_dwd import _core, _estimators, _native_optimizer, _training_kernels

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None

requires_native = unittest.skipIf(_sgd_accel is None, 'Optional compiled extension unavailable')


def problem(m=13):
    x = np.linspace(-1.5, 1.5, m)
    return np.column_stack((x, .13 * np.cos(np.arange(m)), x * x / 7)), np.where(x < 0, -1, 1)


@contextmanager
def factory_spy():
    original = _training_kernels.make_rbf_column_loader
    spy = Mock(wraps=original)
    with ExitStack() as stack:
        stack.enter_context(patch.object(_training_kernels, 'make_rbf_column_loader', spy))
        # Support either a module-qualified call or an imported private alias.
        if getattr(_native_optimizer, 'make_rbf_column_loader', None) is original:
            stack.enter_context(patch.object(_native_optimizer, 'make_rbf_column_loader', spy))
        yield spy


def linear_kernel(X, Y):
    return X @ Y.T


class CustomGenerator(np.random.Generator):
    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self.bit_generator, memo))

    def integers(self, *args, **kwargs):
        return super().integers(*args, **kwargs)


class CustomPCG64(np.random.PCG64):
    pass


class RBFColumnTests(unittest.TestCase):
    def exact(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_column_bytes_match_original_orientation_and_inputs_never_change(self):
        ordinary, _ = problem(7)
        duplicates = np.vstack((ordinary[:3], ordinary[:3], np.zeros((1, 3))))
        offset = np.array([[1e150, -1e150], [1e150 + 1e140, -1e150],
                           [1e150, -1e150 + 1e140], [1e150 + 2e140, -1e150]])
        readonly = ordinary.copy()
        readonly.flags.writeable = False
        cases = [(ordinary, .23), (duplicates, .23), (readonly, .23),
                 (np.asfortranarray(ordinary), .23), (ordinary[:, ::2], .23),
                 (ordinary, float(np.nextafter(0., 1.))), (ordinary, 1e5), (offset, 1e-280)]
        for X, gamma in cases:
            with self.subTest(shape=X.shape, gamma=gamma, writable=X.flags.writeable):
                before = X.tobytes()
                loader = _training_kernels.make_rbf_column_loader(X, gamma)
                columns = []
                for index in range(len(X)):
                    result = loader(index)
                    expected = _core.rbf_kernel(X, X[index:index + 1], gamma)[:, 0]
                    self.exact(result, expected)
                    self.assertTrue(result.flags.c_contiguous)
                    self.assertFalse(np.shares_memory(result, X))
                    for previous in columns:
                        self.assertFalse(np.shares_memory(previous, result))
                    columns.append(result)
                before_again = [column.tobytes() for column in columns]
                loader(0)
                self.assertEqual([column.tobytes() for column in columns], before_again)
                self.assertEqual(X.tobytes(), before)

    def test_factory_validates_features_and_width_once_not_each_column(self):
        X, _ = problem(7)
        array = _core._finite_array
        scalar = _core._finite_scalar
        with patch.object(_core, '_finite_array', wraps=array) as arrays, \
                patch.object(_core, '_finite_scalar', wraps=scalar) as scalars:
            loader = _training_kernels.make_rbf_column_loader(X, .2)
            initial_arrays, initial_scalars = arrays.call_count, scalars.call_count
            self.assertGreaterEqual(initial_arrays, 1)
            self.assertEqual(initial_scalars, 1)
            for index in (0, 3, 0, 6, 1):
                loader(index)
            self.assertEqual(arrays.call_count, initial_arrays)
            self.assertEqual(scalars.call_count, initial_scalars)

    def test_cdist_receives_original_left_by_right_shape_and_metric(self):
        X, _ = problem(7)
        original = _core.cdist
        requests = []

        def trace(left, right, *args, **kwargs):
            requests.append((left.shape, right.shape, right.copy(), args, kwargs.copy()))
            return original(left, right, *args, **kwargs)

        with patch.object(_core, 'cdist', side_effect=trace):
            loader = _training_kernels.make_rbf_column_loader(X, .2)
            loader(3)
        self.assertEqual(len(requests), 1)
        left_shape, right_shape, right, args, kwargs = requests[0]
        self.assertEqual(left_shape, X.shape)
        self.assertEqual(right_shape, (1, X.shape[1]))
        self.exact(right, X[3:4])
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {'metric': 'sqeuclidean'})

    def test_distance_and_multiplication_overflow_keep_original_failures(self):
        cases = [(np.array([[1e308], [-1e308]]), .2),
                 (np.array([[0.], [2.]]), float(np.finfo(np.float64).max))]
        for X, gamma in cases:
            with self.subTest(X=X.tolist(), gamma=gamma):
                before = X.tobytes()
                with self.assertRaises(FloatingPointError) as reference:
                    _core.rbf_kernel(X, X[:1], gamma)
                loader = _training_kernels.make_rbf_column_loader(X, gamma)
                with self.assertRaises(FloatingPointError) as candidate:
                    loader(0)
                self.assertEqual(str(candidate.exception), str(reference.exception))
                self.assertEqual(X.tobytes(), before)

    def test_final_kernel_nonfinite_check_is_retained(self):
        X, _ = problem(7)
        loader = _training_kernels.make_rbf_column_loader(X, .2)

        def broken_exp(values, out):
            out[:] = np.nan
            return out

        with patch.object(np, 'exp', side_effect=broken_exp):
            with self.assertRaisesRegex(FloatingPointError, 'RBF kernel'):
                loader(0)

    def test_factory_rejects_bad_features_and_widths_before_returning_loader(self):
        X, _ = problem(7)
        bad_features = [np.array([[np.nan], [0.]]), np.array([[np.inf], [0.]]),
                        np.ones((2, 0)), np.ones(3), np.ones((2, 3), dtype=complex) * 1j]
        for value in bad_features:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                before = value.tobytes()
                with self.assertRaises(TypeError if value.dtype != np.float64 else ValueError):
                    _training_kernels.make_rbf_column_loader(value, .2)
                self.assertEqual(value.tobytes(), before)
        for gamma in (0., -1., float('inf'), float('nan'), 1j, [1., 2.]):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    _training_kernels.make_rbf_column_loader(X, gamma)

    def test_invalid_training_features_reject_even_when_sklearn_assumes_finite(self):
        X, y = problem(7)
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                bad = X.copy()
                bad[1, 0] = value
                before = bad.tobytes()
                with config_context(assume_finite=True), factory_spy() as factory:
                    with self.assertRaises(ValueError):
                        KernelSGDDWD(max_epochs=1, optimizer_backend='auto').fit(bad, y)
                factory.assert_not_called()
                self.assertEqual(bad.tobytes(), before)


@requires_native
class NativeRBFFitTests(unittest.TestCase):
    def assert_same_fit(self, a, b):
        for name in ('initial_state_', 'final_state_', 'selected_state_', 'rng_state_',
                     'selected_iteration_', 'n_updates_', 'n_samples_seen_', 'learning_rate_B_',
                     'stop_reason_', 'model_status_', 'fit_usable_', 'cv_eligible_',
                     'kernel_evaluation_count_', 'kernel_total_evaluation_count_',
                     'kernel_preparation_evaluation_count_', 'kernel_cache_hits_',
                     'kernel_cache_misses_', 'kernel_cache_evictions_', 'kernel_cache_peak_bytes_',
                     'kernel_cache_capacity_columns_', 'coef_', 'dual_coef_', 'intercept_',
                     'stage_seeds_', 'selected_indices_', 'selected_ids_', 'selected_class_counts_',
                     'selection_scores_'):
            if hasattr(a, name):
                self.assertEqual(pickle.dumps(getattr(a, name), protocol=5),
                                 pickle.dumps(getattr(b, name), protocol=5), name)
        if isinstance(a, TwoStageDWDClassifier):
            self.assert_same_fit(a.linear_estimator_, b.linear_estimator_)
            self.assert_same_fit(a.kernel_estimator_, b.kernel_estimator_)

    def test_full_fit_bytes_and_counts_match_python_for_all_cache_budgets(self):
        X, y = problem(13)
        for sampling in ('with_replacement', 'shuffle'):
            for capacity in (0, len(y) * 8 - 1, len(y) * 8 * 2, len(y) ** 2 * 8):
                for gamma in (.2, 'scale', 'auto'):
                    with self.subTest(sampling=sampling, capacity=capacity, gamma=gamma):
                        rng = np.random.Generator(np.random.Philox(2017))
                        external = pickle.dumps(rng, protocol=5)
                        options = dict(max_epochs=3, random_state=rng, sampling=sampling,
                                       gamma=gamma, kernel_cache_bytes=capacity)
                        reference = KernelSGDDWD(**options).fit(X, y)
                        with factory_spy() as factory:
                            candidate = KernelSGDDWD(optimizer_backend='native', **options).fit(X, y)
                        factory.assert_called_once()
                        self.assert_same_fit(reference, candidate)
                        self.assertEqual(reference.decision_function(X).tobytes(),
                                         candidate.decision_function(X).tobytes())
                        self.assertEqual(pickle.dumps(rng, protocol=5), external)

    def test_two_stage_screening_and_predictions_match_python(self):
        X, y = problem(14)
        for sampling in ('with_replacement', 'shuffle'):
            with self.subTest(sampling=sampling):
                options = dict(retention_fraction=.8, max_epochs=3, random_state=2017,
                               sampling=sampling, gamma='scale', kernel_cache_bytes=14 * 14 * 8)
                reference = TwoStageDWDClassifier(**options).fit(X, y)
                with factory_spy() as factory:
                    candidate = TwoStageDWDClassifier(optimizer_backend='native', **options).fit(X, y)
                factory.assert_called_once()
                self.assert_same_fit(reference, candidate)
                self.assertEqual(reference.decision_function(X).tobytes(),
                                 candidate.decision_function(X).tobytes())

    def test_zero_other_kernels_and_python_only_controls_keep_original_routes(self):
        X, y = problem(13)
        cases = [({'gamma': 0.}, {}), ({'kernel': 'linear'}, {}),
                 ({'kernel': linear_kernel}, {}), ({'optimizer_backend': 'python'}, {}),
                 ({'batch_size': 2}, {}), ({'max_seconds': 100.}, {}),
                 ({}, {'checkpoint_callback': lambda *args: None}),
                 ({}, {'validation_data': (X, y)})]
        for settings, fit_options in cases:
            with self.subTest(settings=settings, fit_options=tuple(fit_options)):
                options = dict(max_epochs=2, random_state=2017, optimizer_backend='auto')
                options.update(settings)
                with factory_spy() as factory:
                    model = KernelSGDDWD(**options).fit(X, y, **fit_options)
                factory.assert_not_called()
                self.assertTrue(model.fit_usable_)

    def test_custom_rngs_use_legacy_loader(self):
        X, y = problem(13)
        generators = (CustomGenerator(np.random.PCG64(17)),
                      np.random.Generator(CustomPCG64(17)))
        for rng in generators:
            with self.subTest(generator=type(rng).__name__, bit=type(rng.bit_generator).__name__):
                external = pickle.dumps(rng, protocol=5)
                with factory_spy() as factory:
                    native = KernelSGDDWD(max_epochs=2, random_state=rng,
                                         optimizer_backend='native').fit(X, y)
                factory.assert_not_called()
                reference = KernelSGDDWD(max_epochs=2, random_state=rng).fit(X, y)
                self.assert_same_fit(reference, native)
                self.assertEqual(pickle.dumps(rng, protocol=5), external)

    def test_replaced_kernel_hooks_are_honored_and_disable_fast_loader(self):
        X, y = problem(13)
        hooks = [(_estimators, '_kernel_values'), (_estimators, 'rbf_kernel'), (_core, 'rbf_kernel')]
        for module, name in hooks:
            with self.subTest(module=module.__name__, hook=name):
                original = getattr(module, name)
                with patch.object(module, name, wraps=original) as hook, factory_spy() as factory:
                    model = KernelSGDDWD(max_epochs=2, random_state=2017,
                                         optimizer_backend='native').fit(X, y)
                factory.assert_not_called()
                if module is _estimators:
                    self.assertEqual(hook.call_count, 2 * len(y))
                self.assertTrue(model.fit_usable_)

    def test_replacing_both_rbf_aliases_with_same_hook_keeps_legacy_evaluation(self):
        X, y = problem(13)
        hook = Mock(wraps=_core.rbf_kernel)
        with patch.object(_estimators, 'rbf_kernel', hook), \
                patch.object(_core, 'rbf_kernel', hook), factory_spy() as factory:
            model = KernelSGDDWD(max_epochs=2, random_state=2017,
                                 optimizer_backend='native').fit(X, y)
        factory.assert_not_called()
        self.assertEqual(hook.call_count, 2 * len(y))
        self.assertTrue(model.fit_usable_)

    def test_fast_cdist_failures_preserve_python_prefix_rng_and_counters(self):
        X, y = problem(13)
        original_cdist = _core.cdist
        for sampling in ('with_replacement', 'shuffle'):
            for capacity in (0, len(y) * 8 - 1, len(y) * 8 * 2, len(y) ** 2 * 8):
                for failed_call in (1, 3):
                    with self.subTest(sampling=sampling, capacity=capacity, failure=failed_call):
                        rng = np.random.Generator(np.random.PCG64DXSM(191))
                        external = pickle.dumps(rng, protocol=5)
                        models = []
                        for backend in ('python', 'native'):
                            calls = [0]

                            def fail_distance(*args, **kwargs):
                                calls[0] += 1
                                if calls[0] == failed_call:
                                    raise FloatingPointError('Controlled direct-distance failure')
                                return original_cdist(*args, **kwargs)

                            model = KernelSGDDWD(max_epochs=3, random_state=rng, sampling=sampling,
                                                 kernel_cache_bytes=capacity, optimizer_backend=backend)
                            with patch.object(_core, 'cdist', side_effect=fail_distance), factory_spy() as factory:
                                with self.assertRaises(NumericalFailure):
                                    model.fit(X, y)
                            self.assertEqual(factory.call_count, int(backend == 'native'))
                            self.assertEqual(calls[0], failed_call)
                            self.assertEqual(model.kernel_evaluation_count_, (failed_call - 1) * len(y))
                            self.assertFalse(model.fit_usable_)
                            self.assertFalse(model.cv_eligible_)
                            self.assertEqual(pickle.dumps(rng, protocol=5), external)
                            models.append(model)
                        self.assert_same_fit(*models)


if __name__ == '__main__':
    unittest.main()
