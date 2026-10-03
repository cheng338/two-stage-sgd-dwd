"""Stopping, usability and checkpoint diagnostics without optimality claims."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from two_stage_sgd_dwd import (KernelSGDDWD, LinearSGDDWD, NumericalFailure,
                               SelectionError, TrainingError, TwoStageDWDClassifier)
from two_stage_sgd_dwd import _estimators


def problem():
    positive = np.column_stack((np.linspace(0.2, 2.0, 20),
                                0.15 * np.sin(np.arange(20))))
    return np.vstack((-positive, positive)), np.repeat([-1, 1], 20)


class FitDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = problem()

    def models(self, **kwargs):
        return [LinearSGDDWD(**kwargs), KernelSGDDWD(**kwargs),
                TwoStageDWDClassifier(retention_fraction=1.0, **kwargs)]

    def stages(self, model):
        if isinstance(model, TwoStageDWDClassifier):
            return [model.linear_estimator_, model.kernel_estimator_]
        return [model]

    def test_fixed_budget_is_usable_without_an_optimality_certificate(self):
        for model in self.models(max_epochs=3, batch_size=7, random_state=11):
            with self.subTest(estimator=type(model).__name__):
                model.fit(self.X, self.y)
                self.assertTrue(model.fit_usable_)
                self.assertTrue(model.cv_eligible_)
                self.assertTrue(model.cap_reached_)
                self.assertTrue(model.criterion_reached_)
                self.assertEqual(model.termination_criterion_, "fixed_epochs")
                self.assertEqual(model.termination_reason_, "epoch_cap")
                self.assertFalse(np.any(model.optimality_checked_))
                self.assertFalse(np.any(model.optimality_met_))
                self.assertFalse(np.any(model.converged_))
                for stage in self.stages(model):
                    self.assertEqual(stage.n_iter_, 18)  # Six minibatches per epoch.
                    self.assertEqual(stage.returned_iteration_, stage.n_updates_)
                    self.assertEqual(stage.n_epochs_completed_, 3.0)
                    self.assertEqual(stage.checkpoint_selection_, "final_iterate")
                    self.assertEqual(stage.stopping_, "fixed")

    def test_auto_without_validation_exactly_matches_explicit_fixed(self):
        for prototype in self.models(max_epochs=3, batch_size=3, random_state=17):
            with self.subTest(estimator=type(prototype).__name__):
                automatic = clone(prototype).fit(self.X, self.y)
                fixed = clone(prototype).set_params(stopping="fixed").fit(self.X, self.y)
                assert_array_equal(automatic.decision_function(self.X), fixed.decision_function(self.X))
                for a, b in zip(self.stages(automatic), self.stages(fixed)):
                    assert_array_equal(a.final_state_["coefficients"], b.final_state_["coefficients"])
                    self.assertEqual(a.rng_state_, b.rng_state_)
                if isinstance(fixed, TwoStageDWDClassifier):
                    assert_array_equal(automatic.selected_indices_, fixed.selected_indices_)
                    self.assertEqual(automatic.stage_seeds_, fixed.stage_seeds_)

    def test_validation_patience_meets_only_the_validation_criterion(self):
        for model in self.models(max_epochs=8, validation_interval=1, patience=2,
                                 min_epochs=0, random_state=7, stopping="validation"):
            with self.subTest(estimator=type(model).__name__):
                with patch.object(_estimators, "dwd_loss", side_effect=lambda x: np.ones_like(x)):
                    model.fit(self.X, self.y, validation_data=(self.X, self.y))
                self.assertEqual(model.termination_reason_, "validation_patience")
                self.assertEqual(model.termination_criterion_, "validation_patience")
                self.assertTrue(model.criterion_reached_)
                self.assertFalse(model.cap_reached_)
                self.assertTrue(model.fit_usable_)
                self.assertTrue(model.cv_eligible_)
                self.assertFalse(np.any(model.converged_))
                self.assertFalse(np.any(model.optimality_checked_))
                for stage in self.stages(model):
                    self.assertEqual(stage.n_epochs_completed_, 3.0)
                    self.assertEqual(stage.n_iter_, 3 * len(self.X))
                    self.assertEqual(stage.returned_iteration_, len(self.X))
                    self.assertEqual(stage.checkpoint_selection_, "validation_loss")

    def test_auto_with_validation_matches_explicit_validation(self):
        settings = dict(max_epochs=6, validation_interval=1, patience=2, min_epochs=0,
                        random_state=13, batch_size="full")
        for prototype in self.models(**settings):
            with self.subTest(estimator=type(prototype).__name__):
                with patch.object(_estimators, "dwd_loss", side_effect=lambda x: np.ones_like(x)):
                    automatic = clone(prototype).fit(self.X, self.y, validation_data=(self.X, self.y))
                    explicit = clone(prototype).set_params(stopping="validation").fit(
                        self.X, self.y, validation_data=(self.X, self.y))
                assert_array_equal(automatic.decision_function(self.X), explicit.decision_function(self.X))
                assert_array_equal(automatic.n_iter_, explicit.n_iter_)
                assert_array_equal(automatic.returned_iteration_, explicit.returned_iteration_)

    def test_validation_epoch_cap_remains_eligible_but_patience_not_reached(self):
        for model in self.models(max_epochs=1, validation_interval=5, random_state=7,
                                 stopping="validation"):
            with self.subTest(estimator=type(model).__name__):
                model.fit(self.X, self.y, validation_data=(self.X, self.y))
                self.assertEqual(model.termination_criterion_, "validation_patience")
                self.assertFalse(model.criterion_reached_)
                self.assertTrue(model.cap_reached_)
                self.assertTrue(model.fit_usable_)
                self.assertTrue(model.cv_eligible_)
                for stage in self.stages(model):
                    self.assertFalse(stage.validation_available_)
                    self.assertEqual(stage.checkpoint_selection_, "final_iterate")

    def test_fixed_disables_patience_but_preserves_independent_checkpoint_selection(self):
        for constructor in (LinearSGDDWD, KernelSGDDWD):
            with self.subTest(estimator=constructor.__name__):
                model = constructor(max_epochs=4, validation_interval=1, patience=1,
                                    min_epochs=0, random_state=7, stopping="fixed")
                values = iter([0.8, 1.0, 0.9, 1.1])
                with patch.object(_estimators, "dwd_loss",
                                  side_effect=lambda x: np.full(len(x), next(values))):
                    model.fit(self.X, self.y, validation_data=(self.X, self.y))
                self.assertEqual(model.n_iter_, 4 * len(self.X))
                self.assertEqual(model.returned_iteration_, len(self.X))
                self.assertEqual(model.checkpoint_selection_, "validation_loss")
                self.assertEqual(model.termination_reason_, "epoch_cap")
                self.assertTrue(model.criterion_reached_)

    def test_partial_minibatch_time_cap_usable_but_not_cv_eligible(self):
        for constructor, step_name in ((LinearSGDDWD, "linear_batch_step"),
                                       (KernelSGDDWD, "kernel_batch_step")):
            with self.subTest(estimator=constructor.__name__):
                clock = [0.0]
                original = getattr(_estimators, step_name)
                def timed_step(*args):
                    result = original(*args)
                    clock[0] += 0.1
                    return result
                model = constructor(max_epochs=3, batch_size=7, random_state=19,
                                    max_seconds=0.15)
                with patch.object(_estimators.time, "perf_counter", side_effect=lambda: clock[0]), \
                        patch.object(_estimators, step_name, side_effect=timed_step):
                    model.fit(self.X, self.y)
                self.assertEqual(model.n_iter_, 2)
                self.assertEqual(model.n_epochs_completed_, 14 / 40)
                self.assertTrue(model.fit_usable_)
                self.assertFalse(model.cv_eligible_)
                self.assertTrue(model.time_limit_reached_)
                self.assertFalse(model.iteration_limit_reached_)
                self.assertFalse(model.criterion_reached_)
                self.assertTrue(np.isfinite(model.decision_function(self.X)).all())

    def test_zero_update_fallback_is_not_usable_or_cv_eligible(self):
        for constructor in (LinearSGDDWD, KernelSGDDWD):
            with self.subTest(estimator=constructor.__name__):
                model = constructor(max_seconds=1e-15).fit(self.X, self.y)
                self.assertFalse(model.fit_usable_)
                self.assertFalse(model.cv_eligible_)
                self.assertEqual(model.n_iter_, 0)
                self.assertEqual(model.n_epochs_completed_, 0.0)
                self.assertTrue(model.cap_reached_)
                self.assertFalse(model.criterion_reached_)

    def test_numerical_failure_keeps_work_diagnostics_not_a_usable_checkpoint(self):
        original = _estimators.linear_step
        calls = [0]
        def fail_after_epoch(*args):
            calls[0] += 1
            if calls[0] > len(self.X):
                raise FloatingPointError("post-checkpoint failure")
            return original(*args)
        model = LinearSGDDWD(max_epochs=3, validation_interval=1, random_state=7)
        with patch.object(_estimators, "linear_step", side_effect=fail_after_epoch):
            with self.assertRaises(NumericalFailure):
                model.fit(self.X, self.y, validation_data=(self.X, self.y))
        self.assertEqual(model.n_iter_, len(self.X))
        self.assertEqual(model.returned_iteration_, len(self.X))
        self.assertEqual(model.termination_reason_, "numerical_failure")
        self.assertFalse(model.fit_usable_)
        self.assertFalse(model.cv_eligible_)
        self.assertFalse(model.criterion_reached_)
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_linear_local_time_cap_is_propagated_after_successful_kernel_fit(self):
        clock = [0.0]
        original = _estimators.linear_step
        def timed_step(*args):
            result = original(*args)
            clock[0] += 0.1
            return result
        model = TwoStageDWDClassifier(retention_fraction=1.0, max_epochs=1,
                                      linear_max_seconds=0.15, random_state=17)
        with patch.object(_estimators.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch.object(_estimators, "linear_step", side_effect=timed_step):
            model.fit(self.X, self.y)
        self.assertEqual(model.stop_reason_, "epoch_cap")  # Existing final-stage alias.
        self.assertTrue(model.fit_usable_)
        self.assertFalse(model.cv_eligible_)
        self.assertTrue(model.time_limit_reached_)
        self.assertFalse(model.criterion_reached_)
        assert_array_equal(model.n_iter_, [2, len(self.X)])
        assert_array_equal(model.n_epochs_completed_, [2 / len(self.X), 1.0])
        self.assertEqual(model.stage_diagnostics_["linear"]["termination_reason"], "time_cap_no_validation")
        self.assertEqual(model.stage_diagnostics_["kernel"]["termination_reason"], "epoch_cap")
        self.assertTrue(model.stage_diagnostics_["kernel"]["cv_eligible"])
        model.stage_diagnostics_["kernel"]["n_iter"] = -1
        self.assertEqual(model.kernel_estimator_.n_iter_, len(self.X))

    def test_two_stage_failures_preserve_available_stage_diagnostics(self):
        model = TwoStageDWDClassifier(retention_fraction=0.01, max_epochs=1, random_state=7)
        with self.assertRaises(SelectionError):
            model.fit(self.X, self.y)
        self.assertTrue(model.stage_diagnostics_["linear"]["fit_usable"])
        self.assertIsNone(model.stage_diagnostics_["kernel"])
        self.assertFalse(model.fit_usable_)
        self.assertFalse(model.cv_eligible_)
        self.assertFalse(model.criterion_reached_)
        assert_array_equal(model.n_iter_, [len(self.X), 0])
        model = TwoStageDWDClassifier(linear_max_seconds=1e-15, random_state=7)
        with self.assertRaises(TrainingError):
            model.fit(self.X, self.y)
        self.assertFalse(model.stage_diagnostics_["linear"]["fit_usable"])
        self.assertIsNone(model.stage_diagnostics_["kernel"])
        self.assertFalse(model.cv_eligible_)

    def test_invalid_stopping_or_missing_validation_invalidates_previous_fit(self):
        for model in self.models(max_epochs=1, random_state=7):
            for setting in ("objective", None, ["fixed"], "validation"):
                with self.subTest(estimator=type(model).__name__, stopping=setting):
                    model.set_params(stopping="auto").fit(self.X, self.y)
                    self.assertEqual(clone(model).stopping, "auto")
                    model.set_params(stopping=setting)
                    with self.assertRaises(ValueError):
                        model.fit(self.X, self.y)
                    with self.assertRaises(NotFittedError):
                        model.predict(self.X)
                    self.assertFalse(hasattr(model, "cv_eligible_"))


if __name__ == "__main__":
    unittest.main()
