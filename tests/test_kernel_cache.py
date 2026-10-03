"""Exact lazy column reuse on tiny data; no dataset or performance fits."""

import copy
import pickle
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from two_stage_sgd_dwd import KernelSGDDWD, NumericalFailure, TwoStageDWDClassifier
from two_stage_sgd_dwd import _estimators
from two_stage_sgd_dwd._cache import LazyKernelColumnCache


def problem():
    x = np.linspace(.3, 1.7, 8)
    positive = np.column_stack((x, .2 * np.sin(np.arange(8))))
    return np.vstack((-positive, positive)), np.repeat([-1, 1], 8)


def matrix_linear(X, Y):
    return X @ Y.T


class ExactAssertions(unittest.TestCase):
    def assertExact(self, first, second):
        """Compare stored numerical bytes, including signed zero, recursively."""
        if isinstance(first, np.ndarray):
            self.assertIsInstance(second, np.ndarray)
            self.assertEqual(first.dtype, second.dtype)
            self.assertEqual(first.shape, second.shape)
            self.assertEqual(first.tobytes(), second.tobytes())
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                with self.subTest(field=key):
                    self.assertExact(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assertExact(a, b)
        elif isinstance(first, (float, np.floating)):
            self.assertEqual(np.float64(first).tobytes(), np.float64(second).tobytes())
        else:
            self.assertEqual(first, second)

    def assertSameFit(self, before, after):
        for attribute in ("initial_state_", "final_state_", "selected_state_", "rng_state_",
                          "selected_iteration_", "n_updates_", "n_samples_seen_",
                          "selected_n_samples_seen_", "learning_rate_B_", "stop_reason_",
                          "model_status_", "cv_eligible_", "fit_usable_",
                          "best_validation_loss_", "termination_reason_", "checkpoint_selection_"):
            with self.subTest(attribute=attribute):
                self.assertExact(getattr(before, attribute), getattr(after, attribute))
        histories = [[{key: value for key, value in entry.items() if key != "elapsed_seconds"}
                      for entry in model.history_] for model in (before, after)]
        self.assertExact(*histories)


class LazyColumnStorageTests(ExactAssertions):
    def test_lazy_lookup_and_lru_eviction_obey_payload_cap(self):
        cache = LazyKernelColumnCache(max_bytes=49, column_size=3)
        calls, returned = [], {}
        self.assertEqual(cache.retained_bytes, 0)
        self.assertEqual(cache.diagnostics()["kernel_cache_misses_"], 0)

        def request(index):
            def compute():
                calls.append(index)
                column = np.array([index, index + .125, -index], dtype=np.float64)
                returned[index] = column
                return column
            result = cache.get(index, compute)
            self.assertLessEqual(cache.retained_bytes, 49)
            return result

        first, hit = request(0)
        self.assertFalse(hit)
        request(1)
        again, hit = request(0)  # Protect 0; the next miss must evict 1.
        self.assertTrue(hit)
        self.assertIs(first, again)
        request(2)
        still_first, hit = request(0)
        self.assertTrue(hit)
        self.assertIs(still_first, first)
        request(1)  # 1 was evicted even though 0 was inserted earlier.
        self.assertEqual(calls, [0, 1, 2, 1])
        self.assertEqual(cache.diagnostics(), {
            "kernel_cache_hits_": 2, "kernel_cache_misses_": 4,
            "kernel_cache_evictions_": 2, "kernel_cache_peak_bytes_": 48,
            "kernel_cache_capacity_columns_": 2})
        before = cache.diagnostics()
        cache.clear()
        self.assertEqual(cache.retained_bytes, 0)
        self.assertEqual(cache.diagnostics(), before)
        _, hit = request(0)
        self.assertFalse(hit)
        self.assertEqual(calls, [0, 1, 2, 1, 0])

    def test_less_than_one_column_never_retains_or_reports_hits(self):
        cache = LazyKernelColumnCache(max_bytes=23, column_size=3)
        calls = []

        def compute():
            calls.append(1)
            return np.array([.2, .4, .6], dtype=np.float64)

        first, hit1 = cache.get(0, compute)
        second, hit2 = cache.get(0, compute)
        self.assertFalse(hit1 or hit2)
        self.assertExact(first, second)
        self.assertEqual(len(calls), 2)
        self.assertEqual(cache.retained_bytes, 0)
        self.assertEqual(cache.diagnostics()["kernel_cache_peak_bytes_"], 0)

    def test_failed_miss_does_not_poison_or_evict_a_valid_entry(self):
        cache = LazyKernelColumnCache(max_bytes=24, column_size=3)
        column, _ = cache.get(0, lambda: np.ones(3, dtype=np.float64))

        def fail():
            raise FloatingPointError("column evaluation failed")

        with self.assertRaisesRegex(FloatingPointError, "evaluation failed"):
            cache.get(1, fail)
        again, hit = cache.get(0, lambda: self.fail("Valid cached entry was lost"))
        self.assertTrue(hit)
        self.assertIs(column, again)
        self.assertEqual(cache.diagnostics()["kernel_cache_evictions_"], 0)
        retried, hit = cache.get(1, lambda: np.full(3, .5, dtype=np.float64))
        self.assertFalse(hit)
        self.assertEqual(retried.tolist(), [.5, .5, .5])
        self.assertEqual(cache.diagnostics()["kernel_cache_misses_"], 3)

    def test_storage_does_not_silently_cast_or_reshape_columns(self):
        cache = LazyKernelColumnCache(max_bytes=24, column_size=3)
        for invalid in (np.ones(3, dtype=np.float32), np.ones((3, 1)),
                        np.ones(6)[::2], [1., 1., 1.]):
            with self.subTest(dtype=getattr(invalid, "dtype", None), shape=getattr(invalid, "shape", None)):
                with self.assertRaises(ValueError):
                    cache.get(0, lambda: invalid)
                self.assertEqual(cache.retained_bytes, 0)


class KernelCacheEstimatorTests(ExactAssertions):
    def setUp(self):
        self.X, self.y = problem()
        self.budget = self.X.shape[0] ** 2 * np.dtype(np.float64).itemsize

    def traced_fit(self, cache_bytes, **settings):
        trajectory, checkpoints = [], []
        original = _estimators.kernel_step

        def step(*args):
            result = original(*args)
            trajectory.append((copy.deepcopy(args), copy.deepcopy(result)))
            return result

        def callback(state, metric):
            checkpoints.append((copy.deepcopy(state),
                                {key: value for key, value in metric.items() if key != "elapsed_seconds"}))

        options = dict(max_epochs=4, validation_interval=1, random_state=731,
                       stopping="fixed", kernel_cache_bytes=cache_bytes)
        options.update(settings)
        model = KernelSGDDWD(**options)
        with patch.object(_estimators, "kernel_step", side_effect=step):
            model.fit(self.X, self.y, validation_data=(self.X[::2], self.y[::2]),
                      checkpoint_callback=callback)
        return model, trajectory, checkpoints

    def test_every_update_checkpoint_rng_and_prediction_are_bitwise_unchanged(self):
        for sampling in ("with_replacement", "shuffle"):
            for kernel, gamma, kws in (("rbf", "scale", None), ("rbf", 0., None),
                                       ("linear", "scale", None),
                                       ("poly", .3, {"degree": 2, "coef0": .7})):
                with self.subTest(sampling=sampling, kernel=kernel, gamma=gamma):
                    options = dict(sampling=sampling, kernel=kernel, gamma=gamma, kernel_kws=kws)
                    baseline, baseline_steps, baseline_checks = self.traced_fit(0, **options)
                    cached, cached_steps, cached_checks = self.traced_fit(self.budget, **options)
                    self.assertExact(baseline_steps, cached_steps)
                    self.assertExact(baseline_checks, cached_checks)
                    self.assertSameFit(baseline, cached)
                    self.assertExact(baseline.dual_coef_, cached.dual_coef_)
                    self.assertExact(baseline.intercept_, cached.intercept_)
                    self.assertExact(baseline.decision_function(self.X), cached.decision_function(self.X))
                    self.assertExact(baseline.predict(self.X), cached.predict(self.X))
                    self.assertExact(baseline.objective(self.X, self.y), cached.objective(self.X, self.y))
                    self.assertGreater(cached.kernel_cache_hits_, 0)
                    self.assertLess(cached.kernel_evaluation_count_, baseline.kernel_evaluation_count_)
                    self.assertEqual(baseline.kernel_cache_hits_, 0)
                    self.assertEqual(baseline.kernel_cache_misses_, 0)
                    self.assertFalse(any(isinstance(value, LazyKernelColumnCache)
                                         for value in vars(cached).values()))

    def test_computation_is_one_requested_column_at_a_time_without_gram_prebuild(self):
        events = []
        kernel, step = _estimators._kernel_values, _estimators.kernel_step

        def record_kernel(X, Y, *args):
            events.append(("column", X.shape, Y.shape))
            return kernel(X, Y, *args)

        def record_step(*args):
            events.append(("update", args[3]))
            return step(*args)

        with patch.object(_estimators, "_kernel_values", side_effect=record_kernel), \
                patch.object(_estimators, "kernel_step", side_effect=record_step):
            model = KernelSGDDWD(kernel_cache_bytes=self.budget, max_epochs=2,
                                 sampling="shuffle", random_state=51).fit(self.X, self.y)
        n = len(self.X)
        self.assertEqual([event[0] for event in events[:2 * n]], ["column", "update"] * n)
        self.assertEqual([event[0] for event in events[2 * n:]], ["update"] * n)
        self.assertTrue(all(event[1:] == (self.X.shape, (1, self.X.shape[1]))
                            for event in events if event[0] == "column"))
        self.assertEqual(model.kernel_cache_misses_, n)
        self.assertEqual(model.kernel_cache_hits_, n)
        self.assertEqual(model.kernel_cache_capacity_columns_, n)
        self.assertEqual(model.kernel_cache_peak_bytes_, self.budget)
        self.assertEqual(model.kernel_cache_evictions_, 0)
        self.assertEqual(model.kernel_evaluation_count_, n * n)

    def test_lru_evictions_do_not_change_any_optimizer_update(self):
        baseline, baseline_steps, baseline_checks = self.traced_fit(0)
        cap = 2 * len(self.X) * np.dtype(np.float64).itemsize
        cached, cached_steps, cached_checks = self.traced_fit(cap)
        self.assertExact(baseline_steps, cached_steps)
        self.assertExact(baseline_checks, cached_checks)
        self.assertSameFit(baseline, cached)
        self.assertGreater(cached.kernel_cache_evictions_, 0)
        self.assertEqual(cached.kernel_cache_capacity_columns_, 2)
        self.assertEqual(cached.kernel_cache_peak_bytes_, cap)

    def test_validation_patience_selects_the_identical_checkpoint(self):
        with patch.object(_estimators, "dwd_loss", side_effect=lambda x: np.ones_like(x)):
            baseline, baseline_steps, baseline_checks = self.traced_fit(
                0, stopping="validation", patience=1, min_epochs=0, max_epochs=8)
            cached, cached_steps, cached_checks = self.traced_fit(
                self.budget, stopping="validation", patience=1, min_epochs=0, max_epochs=8)
        self.assertExact(baseline_steps, cached_steps)
        self.assertExact(baseline_checks, cached_checks)
        self.assertSameFit(baseline, cached)
        self.assertEqual(cached.stop_reason_, "validation_patience")
        self.assertEqual(cached.selected_iteration_, len(self.X))
        self.assertEqual(cached.n_updates_, 2 * len(self.X))

    def test_zero_capacity_still_computes_on_demand_and_counts_no_false_hits(self):
        model = KernelSGDDWD(kernel_cache_bytes=1, max_epochs=2, sampling="shuffle",
                             random_state=4).fit(self.X, self.y)
        self.assertEqual(model.kernel_cache_capacity_columns_, 0)
        self.assertEqual(model.kernel_cache_misses_, 2 * len(self.X))
        self.assertEqual(model.kernel_cache_hits_, 0)
        self.assertEqual(model.kernel_cache_peak_bytes_, 0)
        self.assertEqual(model.kernel_cache_evictions_, 0)

    def test_two_stage_selection_seeds_and_both_trajectories_remain_unchanged(self):
        settings = dict(retention_fraction=.75, max_epochs=4, random_state=72,
                        validation_interval=1, stopping="fixed")
        ids = np.array([f"tiny:{i}" for i in range(len(self.X))])

        def fit(budget):
            trajectory = []
            linear, kernel = _estimators.linear_step, _estimators.kernel_step

            def traced(stage, function):
                def step(*args):
                    result = function(*args)
                    trajectory.append((stage, copy.deepcopy(args), copy.deepcopy(result)))
                    return result
                return step

            with patch.object(_estimators, "linear_step", side_effect=traced("linear", linear)), \
                    patch.object(_estimators, "kernel_step", side_effect=traced("kernel", kernel)):
                model = TwoStageDWDClassifier(kernel_cache_bytes=budget, **settings).fit(
                    self.X, self.y, observation_ids=ids, validation_data=(self.X, self.y))
            return model, trajectory

        baseline, baseline_steps = fit(0)
        cached, cached_steps = fit(self.budget)
        self.assertExact(baseline_steps, cached_steps)
        for attribute in ("stage_seeds_", "selected_indices_", "selected_ids_", "selection_scores_",
                          "selected_class_counts_", "gamma_", "kernel_kws_"):
            self.assertExact(getattr(baseline, attribute), getattr(cached, attribute))
        for stage in ("linear_estimator_", "kernel_estimator_"):
            self.assertSameFit(getattr(baseline, stage), getattr(cached, stage))
        self.assertExact(baseline.predict(self.X), cached.predict(self.X))
        self.assertExact(baseline.decision_function(self.X), cached.decision_function(self.X))
        for name in ("hits", "misses", "peak_bytes", "capacity_columns", "evictions"):
            attribute = "kernel_cache_" + name + "_"
            self.assertEqual(getattr(cached, attribute), getattr(cached.kernel_estimator_, attribute))

    def test_refit_starts_empty_and_uses_new_feature_values(self):
        model = KernelSGDDWD(kernel_cache_bytes=self.budget, max_epochs=3,
                             sampling="shuffle", random_state=8).fit(self.X, self.y)
        self.assertGreater(model.kernel_cache_hits_, 0)
        changed = self.X + np.column_stack((np.arange(len(self.X)) / 7, np.zeros(len(self.X))))
        untouched = changed.copy()
        model.set_params(max_epochs=1).fit(changed, self.y)
        fresh = clone(model).set_params(kernel_cache_bytes=0).fit(changed, self.y)
        self.assertSameFit(fresh, model)
        self.assertExact(fresh.dual_coef_, model.dual_coef_)
        self.assertEqual(model.kernel_cache_misses_, len(changed))
        self.assertEqual(model.kernel_cache_hits_, 0)
        self.assertExact(changed, untouched)
        restored = pickle.loads(pickle.dumps(model))
        self.assertExact(model.decision_function(changed), restored.decision_function(changed))

    def test_invalid_cache_budgets_and_unsupported_configs_are_explicit(self):
        for value in (-1, True, 1.5, "1024", None):
            for constructor in (KernelSGDDWD, TwoStageDWDClassifier):
                with self.subTest(constructor=constructor.__name__, budget=value):
                    with self.assertRaisesRegex(ValueError, "kernel_cache_bytes"):
                        constructor(kernel_cache_bytes=value, max_epochs=1).fit(self.X, self.y)
        for settings in ({"batch_size": 2}, {"batch_size": "full"},
                         {"kernel": matrix_linear}, {"kernel": "sigmoid"}):
            with self.subTest(settings=settings):
                model = KernelSGDDWD(kernel_cache_bytes=self.budget, max_epochs=1).fit(self.X, self.y)
                with self.assertRaises(ValueError):
                    model.set_params(**settings).fit(self.X, self.y)
                with self.assertRaises(NotFittedError):
                    model.predict(self.X)

    def test_cache_hits_do_not_bypass_deadlines_or_make_truncated_fit_eligible(self):
        for budget in (0, self.budget):
            with self.subTest(budget=budget):
                clock = [0.]
                original = _estimators.kernel_step

                def timed_step(*args):
                    result = original(*args)
                    clock[0] += .1
                    return result

                model = KernelSGDDWD(kernel_cache_bytes=budget, max_epochs=4,
                                     sampling="shuffle", random_state=7, max_seconds=1.65)
                with patch.object(_estimators.time, "perf_counter", side_effect=lambda: clock[0]), \
                        patch.object(_estimators, "kernel_step", side_effect=timed_step):
                    model.fit(self.X, self.y)
                self.assertEqual(model.n_updates_, 17)  # First repeated column is already cached.
                self.assertTrue(model.time_limit_reached_)
                self.assertFalse(model.cv_eligible_)
                self.assertTrue(model.fit_usable_)
                if budget:
                    self.assertEqual(model.kernel_cache_hits_, 1)

    def test_failure_after_cached_epoch_invalidates_old_validation_checkpoint(self):
        original = _estimators.kernel_step
        calls = [0]

        def fail_on_hit(*args):
            calls[0] += 1
            if calls[0] == len(self.X) + 1:
                raise FloatingPointError("failed after cached column lookup")
            return original(*args)

        model = KernelSGDDWD(kernel_cache_bytes=self.budget, max_epochs=3,
                             sampling="shuffle", validation_interval=1, random_state=7)
        with patch.object(_estimators, "kernel_step", side_effect=fail_on_hit):
            with self.assertRaises(NumericalFailure):
                model.fit(self.X, self.y, validation_data=(self.X, self.y))
        self.assertEqual(model.n_updates_, len(self.X))
        self.assertEqual(model.kernel_cache_hits_, 1)
        self.assertFalse(model.cv_eligible_)
        self.assertFalse(model.fit_usable_)
        self.assertEqual(model.stop_reason_, "numerical_failure")
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_callback_exception_releases_fit_local_cached_arrays(self):
        created = []

        class TracedCache(LazyKernelColumnCache):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

        def callback(state, metric):
            if state["iteration"]:
                self.assertGreater(created[-1].retained_bytes, 0)
                raise RuntimeError("checkpoint callback failed")

        model = KernelSGDDWD(kernel_cache_bytes=self.budget, max_epochs=3,
                             sampling="shuffle", validation_interval=1, random_state=8)
        with patch.object(_estimators, "LazyKernelColumnCache", TracedCache):
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                model.fit(self.X, self.y, validation_data=(self.X, self.y),
                          checkpoint_callback=callback)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].retained_bytes, 0)
        self.assertGreater(created[0].diagnostics()["kernel_cache_peak_bytes_"], 0)
        with self.assertRaises(NotFittedError):
            model.predict(self.X)


if __name__ == "__main__":
    unittest.main()
