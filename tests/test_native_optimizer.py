"""Tiny full-fit native orchestration, dispatch, and failed-prefix checks."""
import copy
import pickle
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from two_stage_sgd_dwd import LinearSGDDWD, KernelSGDDWD, TwoStageDWDClassifier, NumericalFailure
from two_stage_sgd_dwd import _estimators, _native_optimizer
from test_kernel_cache import ExactAssertions, problem, matrix_linear

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None

requires_native = unittest.skipIf(_sgd_accel is None, 'Optional compiled extension unavailable')


class NativeOptimizerTests(ExactAssertions):
    def setUp(self):
        self.X, self.y = problem()

    def stages(self, model):
        return ([model.linear_estimator_, model.kernel_estimator_]
                if isinstance(model, TwoStageDWDClassifier) else [model])

    def model(self, constructor, **settings):
        options = dict(max_epochs=3, random_state=2017)
        if constructor is TwoStageDWDClassifier:
            options['retention_fraction'] = .75
        options.update(settings)
        return constructor(**options)

    def compare_models(self, before, after):
        for a, b in zip(self.stages(before), self.stages(after)):
            self.assertSameFit(a, b)
            for attribute in ('coef_', 'dual_coef_', 'intercept_', 'basis_', 'kernel_evaluation_count_',
                              'kernel_cache_hits_', 'kernel_cache_misses_', 'kernel_cache_peak_bytes_',
                              'kernel_cache_evictions_'):
                if hasattr(a, attribute):
                    self.assertExact(getattr(a, attribute), getattr(b, attribute))
        if isinstance(before, TwoStageDWDClassifier):
            for attribute in ('stage_seeds_', 'selected_indices_', 'selected_ids_',
                              'selected_class_counts_', 'selection_scores_'):
                self.assertExact(getattr(before, attribute), getattr(after, attribute))
        self.assertExact(before.predict(self.X), after.predict(self.X))
        self.assertExact(before.decision_function(self.X), after.decision_function(self.X))

    @requires_native
    def test_all_models_epoch_prefixes_sampling_and_cache_are_bitwise_equal(self):
        for constructor in (LinearSGDDWD, KernelSGDDWD, TwoStageDWDClassifier):
            for sampling in ('with_replacement', 'shuffle'):
                for epochs in (1, 3):
                    budgets = (None,) if constructor is LinearSGDDWD else (0, 256)
                    for budget in budgets:
                        with self.subTest(model=constructor.__name__, sampling=sampling,
                                          epochs=epochs, cache=budget):
                            settings = dict(sampling=sampling, max_epochs=epochs)
                            if budget is not None:
                                settings['kernel_cache_bytes'] = budget
                            baseline = self.model(constructor, **settings).fit(self.X, self.y)
                            native = self.model(constructor, optimizer_backend='native', **settings).fit(self.X, self.y)
                            self.compare_models(baseline, native)
                            self.assertEqual(native.optimizer_backend_, 'native')
                            self.assertTrue(all(stage.optimizer_backend_ == 'native' for stage in self.stages(native)))

    @requires_native
    def test_epoch_requests_preserve_scalar_index_stream_and_learning_rates(self):
        for sampling in ('with_replacement', 'shuffle'):
            with self.subTest(sampling=sampling):
                requests = []
                actual = _sgd_accel.run_steps

                def trace(kind, coefficient, bias, get_row, labels, indices, rates, lambd):
                    requests.append((indices.copy(), rates.copy()))
                    return actual(kind, coefficient, bias, get_row, labels, indices, rates, lambd)

                with patch.object(_sgd_accel, 'run_steps', side_effect=trace):
                    model = self.model(KernelSGDDWD, optimizer_backend='native', sampling=sampling).fit(self.X, self.y)
                rng = np.random.default_rng()
                rng.bit_generator.state = copy.deepcopy(model.initial_state_['rng_state'])
                m = len(self.X)
                for epoch, (indices, rates) in enumerate(requests):
                    expected = (np.asarray(rng.permutation(m), dtype=np.int64) if sampling == 'shuffle' else
                                np.array([int(rng.integers(m)) for _ in range(m)], dtype=np.int64))
                    self.assertExact(indices, expected)
                    expected_rates = np.array([(1 / model.learning_rate_B_) *
                        (1 + (epoch * m + j) / m) ** (-.75) for j in range(m)])
                    self.assertExact(rates, expected_rates)
                self.assertEqual(len(requests), model.max_epochs)
                self.assertExact(model.rng_state_, rng.bit_generator.state)

    @requires_native
    def test_numpy_generator_inputs_are_not_consumed_and_match_python(self):
        for generator in (np.random.default_rng(39), np.random.RandomState(39)):
            with self.subTest(generator=type(generator).__name__):
                before = pickle.dumps(generator)
                settings = dict(random_state=generator, max_epochs=2)
                baseline = self.model(KernelSGDDWD, **settings).fit(self.X, self.y)
                native = self.model(KernelSGDDWD, optimizer_backend='native', **settings).fit(self.X, self.y)
                self.compare_models(baseline, native)
                self.assertEqual(pickle.dumps(generator), before)

    def test_auto_unsupported_paths_fall_back_with_explicit_reason(self):
        cases = [({'batch_size': 2}, {}, 'batch_size_not_one'),
                 ({'max_seconds': 10.}, {}, 'deadline'),
                 ({}, {'checkpoint_callback': lambda *args: None}, 'checkpoint_callback'),
                 ({}, {'validation_data': (self.X, self.y)}, 'validation_data'),
                 ({'kernel': matrix_linear}, {}, 'unsupported_kernel')]
        for settings, fit_options, reason in cases:
            with self.subTest(reason=reason):
                baseline = self.model(KernelSGDDWD, **settings).fit(self.X, self.y, **fit_options)
                automatic = self.model(KernelSGDDWD, optimizer_backend='auto', **settings).fit(
                    self.X, self.y, **fit_options)
                self.compare_models(baseline, automatic)
                self.assertEqual(automatic.optimizer_backend_, 'python')
                self.assertEqual(automatic.optimizer_backend_reason_, reason)
                forced = self.model(KernelSGDDWD, optimizer_backend='native', **settings)
                with self.assertRaisesRegex(ValueError, reason):
                    forced.fit(self.X, self.y, **fit_options)

    def test_noncontiguous_or_unaligned_features_are_not_silently_copied_for_native(self):
        raw = bytearray(self.X.nbytes + 1)
        unaligned = np.ndarray(self.X.shape, dtype=np.float64, buffer=raw, offset=1)
        unaligned[:] = self.X
        for X in (np.asfortranarray(self.X), unaligned):
            with self.subTest(c_contiguous=X.flags.c_contiguous, aligned=X.flags.aligned):
                baseline = self.model(LinearSGDDWD).fit(X, self.y)
                automatic = self.model(LinearSGDDWD, optimizer_backend='auto').fit(X, self.y)
                self.compare_models(baseline, automatic)
                self.assertEqual(automatic.optimizer_backend_reason_, 'noncontiguous_or_unaligned_features')
                with self.assertRaisesRegex(ValueError, 'noncontiguous_or_unaligned_features'):
                    self.model(LinearSGDDWD, optimizer_backend='native').fit(X, self.y)

    def test_unavailable_extension_has_explicit_auto_fallback_and_forced_error(self):
        original = _native_optimizer.importlib.import_module

        def unavailable(name, *args, **kwargs):
            if name == '._sgd_accel':
                raise ImportError('controlled missing extension')
            return original(name, *args, **kwargs)

        baseline = self.model(LinearSGDDWD).fit(self.X, self.y)
        with patch.object(_native_optimizer.importlib, 'import_module', side_effect=unavailable):
            automatic = self.model(LinearSGDDWD, optimizer_backend='auto').fit(self.X, self.y)
            with self.assertRaisesRegex(ImportError, 'optional compiled'):
                self.model(LinearSGDDWD, optimizer_backend='native').fit(self.X, self.y)
        self.compare_models(baseline, automatic)
        self.assertEqual(automatic.optimizer_backend_reason_, 'extension_unavailable')

    @requires_native
    def test_structural_native_error_propagates_without_python_retry(self):
        with patch.object(_sgd_accel, 'run_steps', side_effect=ValueError('native structural failure')), \
                patch.object(_estimators, 'linear_step', side_effect=AssertionError('unexpected fallback')):
            with self.assertRaisesRegex(ValueError, 'native structural failure'):
                self.model(LinearSGDDWD, optimizer_backend='auto').fit(self.X, self.y)

    @requires_native
    def test_numerical_failure_restores_exact_failed_draw_rng_and_last_good_state(self):
        original = _estimators._kernel_values
        for sampling in ('with_replacement', 'shuffle'):
            for failed_call in (1, len(self.X) + 4):
                with self.subTest(sampling=sampling, failed_call=failed_call):
                    models = []
                    for backend in ('python', 'native'):
                        count = [0]

                        def fail_column(*args):
                            count[0] += 1
                            if count[0] == failed_call:
                                raise FloatingPointError('controlled column failure')
                            return original(*args)

                        model = self.model(KernelSGDDWD, optimizer_backend=backend,
                                           sampling=sampling, kernel_cache_bytes=0)
                        with patch.object(_estimators, '_kernel_values', side_effect=fail_column):
                            with self.assertRaises(NumericalFailure):
                                model.fit(self.X, self.y)
                        self.assertEqual(count[0], failed_call)
                        self.assertEqual(model.n_updates_, failed_call - 1)
                        self.assertFalse(model.fit_usable_)
                        self.assertFalse(model.cv_eligible_)
                        with self.assertRaises(NotFittedError):
                            model.predict(self.X)
                        models.append(model)
                    self.assertSameFit(*models)
                    self.assertEqual(models[0].error_, models[1].error_)

    @requires_native
    def test_invalid_backend_refit_clears_fit_and_clone_pickle_preserve_api(self):
        for constructor in (LinearSGDDWD, KernelSGDDWD, TwoStageDWDClassifier):
            with self.subTest(model=constructor.__name__):
                model = self.model(constructor, optimizer_backend='native').fit(self.X, self.y)
                self.assertEqual(clone(model).optimizer_backend, 'native')
                restored = pickle.loads(pickle.dumps(model))
                self.assertExact(model.decision_function(self.X), restored.decision_function(self.X))
                model.set_params(optimizer_backend='python').fit(self.X, self.y)
                self.assertEqual(model.optimizer_backend_, 'python')
                for invalid in ('compiled', None, True, ['native']):
                    with self.subTest(invalid=invalid):
                        with self.assertRaisesRegex(ValueError, 'optimizer_backend'):
                            model.set_params(optimizer_backend=invalid).fit(self.X, self.y)
                        with self.assertRaises(NotFittedError):
                            model.predict(self.X)


if __name__ == '__main__':
    unittest.main()
