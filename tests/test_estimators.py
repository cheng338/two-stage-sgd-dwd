"""Public estimator contracts on small synthetic data; no benchmark data."""

import pickle
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from two_stage_sgd_dwd import (
    KernelSGDDWD,
    LinearSGDDWD,
    NumericalFailure,
    SelectionError,
    TrainingError,
    TwoStageDWDClassifier,
)
from two_stage_sgd_dwd import _estimators


def sample_data():
    """A fixed symmetric two-class problem with nontrivial feature scales."""
    first = np.linspace(0.2, 2.0, 20)
    positive = np.column_stack((first, 0.15 * np.sin(np.arange(20))))
    X = np.vstack((-positive, positive))
    y = np.array(["negative"] * 20 + ["positive"] * 20)
    return X, y


class EstimatorContractTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = sample_data()

    def models(self):
        return [
            LinearSGDDWD(max_epochs=3, random_state=19),
            KernelSGDDWD(max_epochs=3, random_state=19),
            TwoStageDWDClassifier(
                max_epochs=3, random_state=19, retention_fraction=1.0
            ),
        ]

    def test_clone_tags_and_unfitted_prediction(self):
        for model in self.models():
            with self.subTest(estimator=type(model).__name__):
                self.assertTrue(is_classifier(model))
                copied = clone(model)
                self.assertEqual(copied.get_params(), model.get_params())
                with self.assertRaises(NotFittedError):
                    copied.predict(self.X[:2])
                with self.assertRaises(NotFittedError):
                    copied.decision_function(self.X[:2])

    def test_fit_returns_self_shapes_labels_and_score(self):
        for model in self.models():
            with self.subTest(estimator=type(model).__name__):
                self.assertIs(model.fit(self.X, self.y), model)
                scores = model.decision_function(self.X)
                predicted = model.predict(self.X)
                self.assertEqual(scores.shape, (len(self.X),))
                self.assertTrue(np.all(np.isfinite(scores)))
                assert_array_equal(model.classes_, ["negative", "positive"])
                assert_array_equal(
                    predicted, np.where(scores >= 0, "positive", "negative")
                )
                self.assertEqual(model.n_features_in_, self.X.shape[1])
                self.assertEqual(model.score(self.X, self.y), np.mean(predicted == self.y))
                self.assertEqual(model.decision_function(self.X[:1]).shape, (1,))

    def test_arbitrary_large_integer_labels(self):
        labels = np.where(self.y == "positive", 2**50 + 17, -(2**50) - 3)
        for model in self.models():
            with self.subTest(estimator=type(model).__name__):
                model.fit(self.X, labels)
                expected = model.classes_[(model.decision_function(self.X) >= 0).astype(int)]
                assert_array_equal(model.predict(self.X), expected)

    def test_no_automatic_scaling_and_coefficient_contract(self):
        model = LinearSGDDWD(max_epochs=2, random_state=13).fit(self.X, self.y)
        self.assertEqual(model.coef_.shape, (1, self.X.shape[1]))
        self.assertEqual(model.intercept_.shape, (1,))
        expected_B = 4.0 * np.max(np.sum(self.X**2, axis=1) + 1.0)
        self.assertAlmostEqual(model.learning_rate_B_, expected_B)
        self.assertLessEqual(np.linalg.norm(model.coef_), 1.0 + 1e-14)
        assert_allclose(
            model.decision_function(self.X),
            self.X @ model.coef_.ravel() + model.intercept_[0],
            rtol=1e-14,
            atol=1e-14,
        )

    def test_kernel_expansion_uses_coefficients_without_extra_labels(self):
        model = KernelSGDDWD(
            max_epochs=4, lambd=0.0085, gamma=0.7, random_state=11,
            prediction_batch_size=3,
        ).fit(self.X, self.y)
        self.assertEqual(model.dual_coef_.shape, (1, len(self.X)))
        self.assertEqual(model.intercept_.shape, (1,))
        assert_array_equal(model.basis_, self.X)
        self.assertAlmostEqual(model.learning_rate_B_, 8.017)
        query = self.X[[0, 7, 18, 22, 35]] + np.array([0.03, -0.02])
        squared = ((query[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2)
        expected = np.exp(-0.7 * squared) @ model.dual_coef_.ravel() + model.intercept_[0]
        assert_allclose(model.decision_function(query), expected, rtol=2e-13, atol=2e-13)
        model.set_params(prediction_batch_size=1)
        assert_allclose(model.decision_function(query), expected, rtol=2e-13, atol=2e-13)

    def test_same_seed_and_serialization_reproduce_scores(self):
        for prototype in self.models():
            with self.subTest(estimator=type(prototype).__name__):
                first = clone(prototype).fit(self.X, self.y)
                second = clone(prototype).fit(self.X, self.y)
                assert_array_equal(first.decision_function(self.X), second.decision_function(self.X))
                loaded = pickle.loads(pickle.dumps(first))
                assert_array_equal(first.predict(self.X), loaded.predict(self.X))
                assert_array_equal(first.decision_function(self.X), loaded.decision_function(self.X))
                if isinstance(first, TwoStageDWDClassifier):
                    assert_array_equal(first.selected_indices_, second.selected_indices_)
                    self.assertEqual(first.stage_seeds_, second.stage_seeds_)

    def test_inputs_and_global_random_state_are_not_mutated(self):
        before_X, before_y = self.X.copy(), self.y.copy()
        state = np.random.get_state()
        for model in self.models():
            model.fit(self.X, self.y)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        assert_array_equal(self.X, before_X)
        assert_array_equal(self.y, before_y)

    def test_caller_supplied_generators_are_not_consumed(self):
        for supplied_rng in [np.random.default_rng(19), np.random.RandomState(19)]:
            original = pickle.dumps(supplied_rng)
            for prototype in self.models():
                with self.subTest(estimator=type(prototype).__name__, rng=type(supplied_rng).__name__):
                    model = clone(prototype).set_params(random_state=supplied_rng)
                    first = model.fit(self.X, self.y).decision_function(self.X)
                    second = model.fit(self.X, self.y).decision_function(self.X)
                    assert_array_equal(first, second)
                    self.assertEqual(pickle.dumps(supplied_rng), original)

    def test_kernel_owns_training_basis(self):
        X = self.X.copy()
        model = KernelSGDDWD(max_epochs=2, random_state=1).fit(X, self.y)
        expected = model.decision_function(self.X)
        X[:] = 999.0
        assert_array_equal(model.decision_function(self.X), expected)

    def test_validation_and_feature_errors_do_not_leave_stale_predictor(self):
        for model in self.models():
            with self.subTest(estimator=type(model).__name__):
                model.fit(self.X, self.y)
                with self.assertRaises(ValueError):
                    model.predict(np.ones((3, self.X.shape[1] + 1)))
                bad = self.X.copy()
                bad[0, 0] = np.nan
                with self.assertRaises(ValueError):
                    model.fit(bad, self.y)
                with self.assertRaises(NotFittedError):
                    model.predict(self.X)

    def test_rejects_nonfinite_sparse_and_single_class_training(self):
        for prototype in self.models():
            for X, y in [
                (np.full_like(self.X, np.inf), self.y),
                (sparse.csr_matrix(self.X), self.y),
                (self.X, np.ones(len(self.X))),
            ]:
                with self.subTest(estimator=type(prototype).__name__, input_type=type(X).__name__):
                    with self.assertRaises((ValueError, TypeError)):
                        clone(prototype).fit(X, y)

    def test_rejects_incompatible_validation_data(self):
        for prototype in self.models():
            for validation in [
                (np.ones((4, 3)), self.y[:4]),
                (self.X[:4], np.array(["unknown"] * 4)),
                (self.X[:4], self.y[:3]),
            ]:
                with self.subTest(estimator=type(prototype).__name__):
                    with self.assertRaises(ValueError):
                        clone(prototype).fit(self.X, self.y, validation_data=validation)

    def test_invalid_parameters(self):
        cases = [
            LinearSGDDWD(max_epochs=0),
            LinearSGDDWD(validation_interval=0),
            LinearSGDDWD(patience=0),
            LinearSGDDWD(tol=-1),
            LinearSGDDWD(max_seconds=-1),
            KernelSGDDWD(gamma=-1),
            KernelSGDDWD(lambd=-1),
            KernelSGDDWD(prediction_batch_size=0),
            TwoStageDWDClassifier(retention_fraction=0),
            TwoStageDWDClassifier(retention_fraction=1.1),
        ]
        for model in cases:
            with self.subTest(estimator=repr(model)):
                with self.assertRaises((ValueError, TypeError)):
                    model.fit(self.X, self.y)

    def test_pipeline_grid_search_cloning_and_prediction(self):
        pipeline = make_pipeline(
            StandardScaler(), LinearSGDDWD(max_epochs=2, random_state=42)
        )
        search = GridSearchCV(
            pipeline, {"linearsgddwd__max_epochs": [2, 4]},
            cv=StratifiedKFold(2, shuffle=True, random_state=3),
            scoring="accuracy", error_score="raise", n_jobs=1,
        ).fit(self.X, self.y)
        self.assertEqual(search.predict(self.X).shape, self.y.shape)
        self.assertTrue(np.all(np.isfinite(search.cv_results_["mean_test_score"])))


class CheckpointAndCapTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = sample_data()

    def test_epoch_counts_and_independent_snapshots(self):
        for model in [LinearSGDDWD(max_epochs=3, random_state=9),
                      KernelSGDDWD(max_epochs=3, random_state=9)]:
            with self.subTest(estimator=type(model).__name__):
                model.fit(self.X, self.y)
                self.assertEqual(model.n_updates_, 3 * len(self.X))
                self.assertEqual(model.n_epochs_, 3)
                self.assertEqual(model.selected_iteration_, model.n_updates_)
                self.assertEqual(model.initial_state_["iteration"], 0)
                self.assertEqual(model.final_state_["iteration"], model.n_updates_)
                self.assertEqual(model.selected_state_["iteration"], model.selected_iteration_)
                states = [model.initial_state_, model.final_state_, model.selected_state_]
                for state in states:
                    self.assertEqual(np.asarray(state["coefficients"]).ndim, 1)
                    self.assertEqual(np.asarray(state["intercept"]).ndim, 0)
                for i, first in enumerate(states):
                    for second in states[i + 1:]:
                        self.assertFalse(np.shares_memory(first["coefficients"], second["coefficients"]))
                saved_initial = model.initial_state_["coefficients"].copy()
                model.final_state_["coefficients"][:] = 123.0
                assert_array_equal(model.initial_state_["coefficients"], saved_initial)
                self.assertFalse(np.any(model.selected_state_["coefficients"] == 123.0))

    def test_flat_validation_patience_stops_at_twenty_epochs(self):
        # Validation scores are just the tiny, unrestricted intercept. The two
        # margins stay in the affine loss branch, so their mean loss is one.
        X = np.array([[-1000.0], [-800.0], [800.0], [1000.0]])
        y = np.array([-1, -1, 1, 1])
        model = LinearSGDDWD(max_epochs=40, random_state=17).fit(
            X, y, validation_data=(np.zeros((2, 1)), np.array([-1, 1]))
        )
        checks = [row for row in model.history_ if row["validation_loss"] is not None]
        self.assertEqual([row["iteration"] for row in checks], [20, 40, 60, 80])
        self.assertEqual([row["patience_misses"] for row in checks], [0, 1, 2, 3])
        assert_allclose([row["validation_loss"] for row in checks], 1.0, atol=1e-15)
        self.assertEqual(model.n_updates_, 80)
        best = min(checks, key=lambda row: (row["validation_loss"], row["iteration"]))
        self.assertEqual(model.selected_iteration_, best["iteration"])
        self.assertEqual(model.history_[0]["iteration"], 0)
        self.assertIsNone(model.history_[0]["validation_loss"])

    def test_zero_update_cap_is_diagnostic_and_ties_choose_positive_class(self):
        for model in [LinearSGDDWD(max_seconds=1e-15, random_state=1),
                      KernelSGDDWD(max_seconds=1e-15, random_state=1)]:
            with self.subTest(estimator=type(model).__name__):
                model.fit(self.X, self.y)
                self.assertEqual(model.n_updates_, 0)
                self.assertEqual(model.model_status_, "untrained_fallback")
                self.assertEqual(model.stop_reason_, "time_cap_no_validation")
                assert_array_equal(model.decision_function(self.X), np.zeros(len(self.X)))
                assert_array_equal(model.predict(self.X), np.repeat("positive", len(self.X)))

    def test_actual_best_checkpoint_differs_from_patience_reference(self):
        # Improvements smaller than tol still replace the actual best model,
        # but cannot reset the independent patience reference or miss counter.
        losses = iter([1.0, 0.99996, 0.99992, 0.99995])
        def scripted_loss(margins):
            return np.full(len(margins), next(losses))

        model = LinearSGDDWD(max_epochs=40, random_state=6)
        with patch.object(_estimators, "dwd_loss", side_effect=scripted_loss):
            model.fit(self.X, self.y, validation_data=(self.X[:4], self.y[:4]))
        checks = [row for row in model.history_ if row["validation_loss"] is not None]
        self.assertEqual([row["patience_misses"] for row in checks], [0, 1, 2, 3])
        assert_array_equal([row["patience_reference_loss"] for row in checks], [1.0] * 4)
        self.assertEqual(model.selected_iteration_, 15 * len(self.X))
        self.assertEqual(model.n_updates_, 20 * len(self.X))
        self.assertEqual(model.stop_reason_, "validation_patience")
        assert_array_equal(model.coef_.ravel(), model.selected_state_["coefficients"])

    def test_finite_trained_cap_before_validation_preserves_latest_state(self):
        for model, step_name in [
            (LinearSGDDWD(max_seconds=0.25, random_state=1), "linear_step"),
            (KernelSGDDWD(max_seconds=0.25, random_state=1), "kernel_step"),
        ]:
            clock = [0.0]
            original_step = getattr(_estimators, step_name)
            def timed_step(*args):
                result = original_step(*args)
                clock[0] += 0.1
                return result

            # Drive elapsed time by work, not by the number of deadline polls.
            with self.subTest(estimator=type(model).__name__):
                with patch.object(_estimators.time, "perf_counter", side_effect=lambda: clock[0]), \
                     patch.object(_estimators, step_name, side_effect=timed_step):
                    model.fit(self.X, self.y)
                self.assertEqual(model.n_updates_, 3)
                self.assertEqual(model.model_status_, "trained")
                self.assertEqual(model.stop_reason_, "time_cap_no_validation")
                self.assertEqual(model.selected_iteration_, 3)
                self.assertFalse(model.validation_available_)
                self.assertTrue(np.all(np.isfinite(model.decision_function(self.X))))

    def test_last_update_crossing_deadline_records_time_cap(self):
        X, y = np.array([[-1.0], [1.0]]), np.array([-1, 1])
        for model, step_name in [
            (LinearSGDDWD(max_epochs=1, max_seconds=1.0, random_state=1), "linear_step"),
            (KernelSGDDWD(max_epochs=1, max_seconds=1.0, random_state=1), "kernel_step"),
        ]:
            clock = [0.0]
            original_step = getattr(_estimators, step_name)
            def timed_step(*args):
                result = original_step(*args)
                clock[0] += 0.625
                return result

            with self.subTest(estimator=type(model).__name__):
                with patch.object(_estimators.time, "perf_counter", side_effect=lambda: clock[0]), \
                     patch.object(_estimators, step_name, side_effect=timed_step):
                    model.fit(X, y)
                # Both finite updates are retained, but the last crosses the
                # deadline. Exhausting the epoch too must not hide the cap.
                self.assertEqual(model.n_updates_, 2)
                self.assertEqual(model.final_state_["iteration"], 2)
                self.assertEqual(model.model_status_, "trained")
                self.assertEqual(model.stop_reason_, "time_cap_no_validation")
                self.assertAlmostEqual(model.fit_seconds_, 1.25)

    def test_interrupted_validation_includes_elapsed_phase_time_and_kernel_work(self):
        clock = [0.0]
        original_kernel = _estimators.rbf_kernel
        completed_validation_blocks = []
        def timed_kernel(X, Z, gamma):
            values = original_kernel(X, Z, gamma)
            if len(Z) == len(self.X):
                # Training updates request one kernel column; validation
                # requests query blocks against the entire fitting basis.
                completed_validation_blocks.append(len(X))
                clock[0] += 0.75
            return values

        model = KernelSGDDWD(
            validation_interval=1, max_epochs=3, max_seconds=0.5, random_state=7,
            prediction_batch_size=2,
        )
        with patch.object(_estimators.time, "perf_counter", side_effect=lambda: clock[0]), \
             patch.object(_estimators, "rbf_kernel", side_effect=timed_kernel):
            model.fit(self.X, self.y, validation_data=(self.X, self.y))
        self.assertEqual(model.n_updates_, len(self.X))
        self.assertEqual(model.stop_reason_, "time_cap_no_validation")
        self.assertFalse(model.validation_available_)
        self.assertEqual(len(model.history_), 1)
        self.assertAlmostEqual(model.validation_checkpoint_seconds_, 0.75)
        self.assertAlmostEqual(model.update_seconds_, 0.0)
        self.assertAlmostEqual(model.optimization_seconds_, 0.75)
        self.assertAlmostEqual(model.fit_seconds_, 0.75)
        self.assertEqual(completed_validation_blocks, [2])
        expected_kernel_work = len(self.X) ** 2 + 2 * len(self.X)
        self.assertEqual(model.kernel_evaluation_count_, expected_kernel_work)

    def test_kernel_fit_time_includes_basis_publication(self):
        clock = [0.0]

        class TimedBasisKernel(KernelSGDDWD):
            def __setattr__(self, name, value):
                if name == "basis_":
                    # Simulate time spent making the owned basis available.
                    # This tests the fit boundary without prescribing which
                    # NumPy allocation/copy primitive an implementation uses.
                    clock[0] += 2.0
                super().__setattr__(name, value)

        for cap in (None, 1.0):
            clock[0] = 0.0
            model = TimedBasisKernel(max_epochs=1, random_state=2, max_seconds=cap)
            with self.subTest(max_seconds=cap):
                with patch.object(_estimators.time, "perf_counter", side_effect=lambda: clock[0]):
                    model.fit(self.X, self.y)
                self.assertAlmostEqual(clock[0], 2.0)
                self.assertAlmostEqual(model.fit_seconds_, 2.0)
                assert_array_equal(model.basis_, self.X)
                if cap is not None:
                    self.assertEqual(model.stop_reason_, "time_cap_no_validation")

    def test_cap_interrupting_validation_never_commits_partial_metric(self):
        model = KernelSGDDWD(validation_interval=1, max_epochs=3, random_state=7)
        with patch.object(_estimators, "_scores", side_effect=_estimators._Deadline):
            model.fit(self.X, self.y, validation_data=(self.X, self.y))
        self.assertEqual(model.n_updates_, len(self.X))
        self.assertEqual(model.stop_reason_, "time_cap_no_validation")
        self.assertEqual(model.model_status_, "trained")
        self.assertFalse(model.validation_available_)
        self.assertEqual(len(model.history_), 1)
        self.assertIsNone(model.history_[0]["validation_loss"])

    def test_numerical_failure_cannot_rescue_older_validated_model(self):
        original_step = _estimators.linear_step
        calls = 0
        def failing_step(*args):
            nonlocal calls
            calls += 1
            if calls > len(self.X):
                raise FloatingPointError("injected post-checkpoint failure")
            return original_step(*args)

        model = LinearSGDDWD(validation_interval=1, max_epochs=3, random_state=5)
        with patch.object(_estimators, "linear_step", side_effect=failing_step):
            with self.assertRaises(NumericalFailure) as caught:
                model.fit(self.X, self.y, validation_data=(self.X, self.y))
        self.assertEqual(caught.exception.status, "numerical_failure")
        self.assertEqual(model.model_status_, "numerical_failure")
        self.assertEqual(model.n_updates_, len(self.X))
        self.assertTrue(model.validation_available_)
        self.assertEqual(model.selected_state_["iteration"], len(self.X))
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_finite_inputs_with_overflowing_learning_bound_record_failure(self):
        # Input validation accepts these finite observations. The subsequent
        # bound computation is a numerical fit failure, not an input NaN.
        model = LinearSGDDWD(max_epochs=1, random_state=5)
        with self.assertRaises(NumericalFailure) as caught:
            model.fit(np.array([[-1e200], [1e200]]), np.array([-1, 1]))
        self.assertEqual(caught.exception.status, "numerical_failure")
        self.assertEqual(model.model_status_, "numerical_failure")
        self.assertEqual(model.stop_reason_, "numerical_failure")
        with self.assertRaises(NotFittedError):
            model.predict(np.zeros((1, 1)))

    def test_mutating_checkpoint_callback_cannot_modify_training(self):
        def destructive_callback(state, metrics):
            state["coefficients"][:] = np.nan
            state["intercept"] = np.inf
            metrics["validation_loss"] = -999.0

        prototype = LinearSGDDWD(max_epochs=10, random_state=4)
        baseline = clone(prototype).fit(self.X, self.y, validation_data=(self.X, self.y))
        monitored = clone(prototype).fit(
            self.X, self.y, validation_data=(self.X, self.y),
            checkpoint_callback=destructive_callback,
        )
        assert_array_equal(monitored.decision_function(self.X), baseline.decision_function(self.X))
        self.assertTrue(all(row["validation_loss"] != -999.0 for row in monitored.history_))

    def test_untrained_linear_stage_cannot_start_two_stage_kernel(self):
        model = TwoStageDWDClassifier(linear_max_seconds=1e-15, random_state=1)
        with self.assertRaises(TrainingError) as caught:
            model.fit(self.X, self.y)
        self.assertEqual(caught.exception.status, "linear_stage_failure")
        with self.assertRaises(NotFittedError):
            model.predict(self.X)


class TwoStageContractTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = sample_data()

    def test_default_retains_thirty_percent_and_explicit_five_percent_remains_available(self):
        prototype = TwoStageDWDClassifier()
        self.assertEqual(prototype.get_params()["retention_fraction"], 0.3)
        self.assertEqual(clone(prototype).retention_fraction, 0.3)
        settings = dict(max_epochs=2, batch_size="full", random_state=7)
        default = clone(prototype).set_params(**settings).fit(self.X, self.y)
        explicit = TwoStageDWDClassifier(retention_fraction=0.3, **settings).fit(self.X, self.y)
        comparison = TwoStageDWDClassifier(retention_fraction=0.05, **settings).fit(self.X, self.y)
        self.assertEqual(len(default.selected_indices_), 12)  # 30% of 40 observations.
        self.assertEqual(len(comparison.selected_indices_), 2)  # Explicit original 5% comparison.
        self.assertEqual(default.kernel_estimator_.basis_.shape[0], 12)
        assert_array_equal(default.selected_indices_, explicit.selected_indices_)
        assert_array_equal(default.decision_function(self.X), explicit.decision_function(self.X))
        self.assertTrue(np.all(np.isin(comparison.selected_indices_, default.selected_indices_)))

    def test_selection_is_exact_global_absolute_rank_and_unique(self):
        ids = np.array([f"train:{i:04}" for i in range(len(self.X))])
        model = TwoStageDWDClassifier(
            max_epochs=4, random_state=7, retention_fraction=0.4
        ).fit(self.X, self.y, observation_ids=ids)
        k = int(np.floor(0.4 * len(self.X)))
        chosen = model.selected_indices_
        self.assertEqual(len(chosen), k)
        self.assertEqual(len(np.unique(chosen)), k)
        assert_array_equal(model.selected_ids_, ids[chosen])
        expected_scores = np.abs(model.linear_estimator_.decision_function(self.X))
        assert_allclose(model.selection_scores_, expected_scores, rtol=0, atol=0)
        remaining = np.setdiff1d(np.arange(len(self.X)), chosen)
        self.assertLessEqual(expected_scores[chosen].max(), expected_scores[remaining].min())
        assert_array_equal(model.kernel_estimator_.basis_, self.X[chosen])
        assert_allclose(model.decision_function(self.X), model.kernel_estimator_.decision_function(self.X))
        counts = model.selected_class_counts_
        count_values = list(counts.values()) if isinstance(counts, dict) else counts
        assert_array_equal(np.sort(count_values), np.sort(np.unique(self.y[chosen], return_counts=True)[1]))

    def test_single_class_selection_is_preserved_failure_not_quota_repair(self):
        ids = np.arange(100, 100 + len(self.X))
        model = TwoStageDWDClassifier(
            max_epochs=2, random_state=42, retention_fraction=0.025
        )
        with self.assertRaises(SelectionError) as caught:
            model.fit(self.X, self.y, observation_ids=ids)
        self.assertEqual(caught.exception.status, "single_class_selection")
        self.assertEqual(len(model.selected_indices_), 1)
        self.assertEqual(len(model.selected_ids_), 1)
        self.assertEqual(model.selection_scores_.shape, (len(self.X),))
        self.assertTrue(np.all(np.isfinite(model.linear_estimator_.decision_function(self.X))))
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_empty_selection_is_preserved_failure_without_kernel_fit(self):
        ids = np.arange(100, 100 + len(self.X))
        model = TwoStageDWDClassifier(
            max_epochs=2, random_state=42, retention_fraction=0.01
        )
        with patch.object(KernelSGDDWD, "fit", side_effect=AssertionError("kernel fit")), \
             self.assertRaises(SelectionError) as caught:
            model.fit(self.X, self.y, observation_ids=ids)
        self.assertEqual(caught.exception.status, "empty_selection")
        self.assertEqual(model.model_status_, "empty_selection")
        self.assertEqual(model.stop_reason_, "selection_failure")
        self.assertFalse(model.cv_eligible_)
        self.assertEqual(model.n_retained_, 0)
        self.assertEqual(len(model.selected_indices_), 0)
        self.assertEqual(len(model.selected_ids_), 0)
        assert_array_equal(model.selected_class_counts_, [0, 0])
        self.assertEqual(model.selection_scores_.shape, (len(self.X),))
        self.assertTrue(np.all(np.isfinite(model.linear_estimator_.decision_function(self.X))))
        self.assertFalse(hasattr(model, "kernel_estimator_"))
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_two_rows_from_one_class_remain_selection_failure(self):
        model = TwoStageDWDClassifier(
            max_epochs=2, random_state=42, retention_fraction=0.05
        )
        # These two adjacent rows have the same original label. Selection must
        # never replace one with a row from the other class merely to fit.
        with patch.object(_estimators, "select_near_boundary", return_value=np.array([0, 1])), \
             patch.object(KernelSGDDWD, "fit", side_effect=AssertionError("kernel fit")), \
             self.assertRaises(SelectionError) as caught:
            model.fit(self.X, self.y)
        self.assertEqual(caught.exception.status, "single_class_selection")
        self.assertEqual(model.n_retained_, 2)
        assert_array_equal(model.selected_indices_, [0, 1])
        assert_array_equal(model.selected_class_counts_, [2, 0])
        self.assertFalse(model.cv_eligible_)
        self.assertFalse(hasattr(model, "kernel_estimator_"))

    def test_observation_ids_must_correspond_to_rows(self):
        model = TwoStageDWDClassifier(max_epochs=1, retention_fraction=1.0)
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y, observation_ids=np.arange(len(self.X) - 1))


if __name__ == "__main__":
    unittest.main()
