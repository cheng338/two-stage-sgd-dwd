"""Estimator batch contracts and independent tiny trajectories.

These are synthetic numerical unit tests, not pilot or dissertation experiments.
The independent formulas deliberately do not call the package's step helpers.
"""

import pickle
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from two_stage_sgd_dwd import (
    KernelSGDDWD,
    LinearSGDDWD,
    SelectionError,
    TwoStageDWDClassifier,
)
from two_stage_sgd_dwd import _estimators


def tiny_data():
    return (
        np.array([[-1.5, 0.2], [-0.7, -0.5], [0.2, 0.6],
                  [0.9, -0.3], [1.4, 0.4]]),
        np.array([-1, -1, 1, 1, 1]),
    )


def selection_data():
    x = np.r_[np.arange(-20.0, 0.0), np.arange(1.0, 21.0)]
    return x[:, None], np.where(x > 0, 1, -1)


def independent_derivative(margins):
    result = np.empty(len(margins))
    for j, margin in enumerate(margins):
        result[j] = -1.0 if margin <= 0.5 else (-0.25 / margin) / margin
    return result


def independent_trajectory(kind, X, y, batch_size, epochs, seed=0,
                           gamma=0.6, lambd=0.035, sampling="with_replacement"):
    """Literal mean-gradient reference, including duplicate sampled indices."""
    m = len(y)
    effective = m if batch_size == "full" else min(int(batch_size), m)
    rng = np.random.default_rng(seed)
    coefficients = np.zeros(X.shape[1] if kind == "linear" else m)
    intercept = 0.0
    B = (4.0 * np.max(np.sum(X * X, axis=1) + 1.0)
         if kind == "linear" else 2.0 * lambd + 8.0)
    K = np.exp(-gamma * ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    seen, records = 0, []
    for _ in range(epochs):
        permutation = (rng.permutation(m)
                       if batch_size != "full" and sampling == "shuffle" else None)
        remaining = m
        while remaining:
            size = min(effective, remaining)
            if batch_size == "full":
                indices = np.arange(m)
            elif sampling == "shuffle":
                indices = permutation[m - remaining:m - remaining + size]
            else:
                indices = rng.integers(m, size=size)
            eta = (1.0 / B) * (1.0 + seen / m) ** (-0.75)
            scores = (X[indices] @ coefficients + intercept if kind == "linear"
                      else K[indices] @ coefficients + intercept)
            q = independent_derivative(y[indices] * scores) * y[indices]
            if kind == "linear":
                proposal = coefficients - eta * np.mean(q[:, None] * X[indices], axis=0)
                coefficients = proposal / max(1.0, float(np.linalg.norm(proposal)))
            else:
                coefficients = (1.0 - eta * (2.0 * lambd)) * coefficients
                # An ordinary loop provides an independent oracle for repeated
                # destinations, which advanced-index '+=' does not accumulate.
                for index, multiplier in zip(indices, q):
                    coefficients[index] -= eta * multiplier / size
            intercept -= eta * float(np.mean(q))
            seen += size
            remaining -= size
            records.append({
                "indices": indices.copy(), "coefficients": coefficients.copy(),
                "intercept": intercept, "samples_seen": seen, "eta": eta,
            })
    return records


class BatchParameterContractTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = tiny_data()

    def models(self, **kwargs):
        parameters = dict(max_epochs=3, random_state=0)
        parameters.update(kwargs)
        return [
            LinearSGDDWD(**parameters),
            KernelSGDDWD(**parameters),
            TwoStageDWDClassifier(retention_fraction=1.0, **parameters),
        ]

    def test_default_and_explicit_one_with_replacement_have_identical_states(self):
        for default in self.models():
            with self.subTest(estimator=type(default).__name__):
                self.assertEqual(default.get_params()["batch_size"], 1)
                self.assertEqual(default.get_params()["sampling"], "with_replacement")
                explicit = clone(default).set_params(batch_size=1, sampling="with_replacement")
                default.fit(self.X, self.y)
                explicit.fit(self.X, self.y)
                assert_array_equal(default.decision_function(self.X), explicit.decision_function(self.X))
                if isinstance(default, TwoStageDWDClassifier):
                    assert_array_equal(default.selected_indices_, explicit.selected_indices_)
                    children = [(default.linear_estimator_, explicit.linear_estimator_),
                                (default.kernel_estimator_, explicit.kernel_estimator_)]
                else:
                    children = [(default, explicit)]
                for first, second in children:
                    assert_array_equal(first.final_state_["coefficients"], second.final_state_["coefficients"])
                    self.assertEqual(first.final_state_["intercept"], second.final_state_["intercept"])
                    self.assertEqual(first.n_updates_, 3 * len(self.X))
                    self.assertEqual(first.n_samples_seen_, 3 * len(self.X))
                    self.assertEqual(first.batch_size_, 1)
                    self.assertEqual(first.sampling_, "with_replacement")

    def test_explicit_batch_one_retains_one_observation_per_update(self):
        for model in self.models(batch_size=1):
            with self.subTest(estimator=type(model).__name__):
                model.fit(self.X, self.y)
                children = ([model.linear_estimator_, model.kernel_estimator_]
                            if isinstance(model, TwoStageDWDClassifier) else [model])
                for child in children:
                    self.assertEqual(child.batch_size_, 1)
                    self.assertEqual(child.n_updates_, 3 * len(self.X))
                    self.assertEqual(child.n_updates_, child.n_samples_seen_)
                    self.assertEqual(child.sampling_, "with_replacement")

    def test_cloning_and_effective_size_do_not_mutate_constructor_parameter(self):
        for requested in (3, np.int64(3), 100, "full"):
            for model in self.models(batch_size=requested):
                with self.subTest(estimator=type(model).__name__, requested=requested):
                    copied = clone(model)
                    self.assertEqual(copied.get_params()["batch_size"], requested)
                    copied.fit(self.X, self.y)
                    self.assertEqual(copied.get_params()["batch_size"], requested)
                    children = ([copied.linear_estimator_, copied.kernel_estimator_]
                                if isinstance(copied, TwoStageDWDClassifier) else [copied])
                    for child in children:
                        expected = len(self.X) if requested == "full" else min(requested, len(self.X))
                        self.assertEqual(child.batch_size_, expected)
                        self.assertEqual(child.sampling_, "full" if requested == "full" else "with_replacement")

    def test_invalid_batch_sizes_are_rejected_by_all_estimators(self):
        for invalid in (0, -1, True, False, np.bool_(True), 1.0, 2.5,
                        None, "2", "FULL", "auto"):
            for model in self.models(batch_size=invalid):
                with self.subTest(estimator=type(model).__name__, batch_size=invalid):
                    with self.assertRaises((TypeError, ValueError)):
                        model.fit(self.X, self.y)

    def test_invalid_sampling_is_rejected_by_all_estimators(self):
        for invalid in (None, True, 1, "replacement", "SHUFFLE", "auto"):
            for model in self.models(batch_size=3, sampling=invalid):
                with self.subTest(estimator=type(model).__name__, sampling=invalid):
                    with self.assertRaises((TypeError, ValueError)):
                        model.fit(self.X, self.y)

    def test_sampling_choice_survives_cloning_and_parameter_assignment(self):
        for requested in ("shuffle", "with_replacement"):
            for model in self.models(batch_size=3, sampling=requested):
                with self.subTest(estimator=type(model).__name__, sampling=requested):
                    copied = clone(model)
                    self.assertEqual(copied.get_params()["sampling"], requested)
                    copied.fit(self.X, self.y)
                    children = ([copied.linear_estimator_, copied.kernel_estimator_]
                                if isinstance(copied, TwoStageDWDClassifier) else [copied])
                    for child in children:
                        self.assertEqual(child.sampling_, requested)

    def test_larger_integer_batch_is_clamped_but_remains_random_sampling(self):
        for kind, constructor in (("linear", LinearSGDDWD), ("kernel", KernelSGDDWD)):
            options = {} if kind == "linear" else {"gamma": 0.6, "lambd": 0.035}
            model = constructor(batch_size=100, sampling="with_replacement",
                                max_epochs=2, random_state=0, **options).fit(self.X, self.y)
            reference = independent_trajectory(kind, self.X, self.y, 100, 2,
                                               sampling="with_replacement")
            coefficients = model.coef_.ravel() if kind == "linear" else model.dual_coef_.ravel()
            with self.subTest(estimator=constructor.__name__):
                self.assertEqual(model.batch_size_, len(self.X))
                self.assertEqual(model.sampling_, "with_replacement")
                self.assertEqual(model.n_updates_, 2)
                self.assertEqual(model.n_samples_seen_, 2 * len(self.X))
                assert_allclose(coefficients, reference[-1]["coefficients"], rtol=2e-13, atol=2e-14)
                self.assertAlmostEqual(model.intercept_[0], reference[-1]["intercept"], places=13)

    def test_minibatches_do_not_mutate_inputs_or_global_rng(self):
        X_before, y_before = self.X.copy(), self.y.copy()
        global_before = pickle.dumps(np.random.get_state())
        for model in self.models(batch_size=3):
            model.fit(self.X, self.y)
        assert_array_equal(self.X, X_before)
        assert_array_equal(self.y, y_before)
        self.assertEqual(pickle.dumps(np.random.get_state()), global_before)


class IndependentBatchTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = tiny_data()

    def test_minibatch_linear_updates_match_mean_gradient_and_sample_schedule(self):
        for sampling in ("shuffle", "with_replacement"):
            with self.subTest(sampling=sampling):
                model = LinearSGDDWD(batch_size=3, sampling=sampling, max_epochs=3,
                                     random_state=0).fit(self.X, self.y)
                reference = independent_trajectory("linear", self.X, self.y, 3, 3,
                                                   sampling=sampling)
                self.assertEqual([len(row["indices"]) for row in reference], [3, 2, 3, 2, 3, 2])
                self.assertEqual(model.n_updates_, 6)
                self.assertEqual(model.n_samples_seen_, 15)
                self.assertEqual(model.n_epochs_, 3)
                self.assertEqual(model.n_updates_per_epoch_, 2)
                assert_allclose(model.coef_.ravel(), reference[-1]["coefficients"], rtol=2e-13, atol=2e-14)
                self.assertAlmostEqual(model.intercept_[0], reference[-1]["intercept"], places=13)
                self.assertLessEqual(np.linalg.norm(model.coef_), 1.0 + 1e-14)
                self.assertEqual(model.final_state_["n_samples_seen"], 15)
                self.assertEqual(model.selected_n_samples_seen_, 15)

    def test_minibatch_kernel_updates_accumulate_duplicate_indices(self):
        model = KernelSGDDWD(
            batch_size=3, max_epochs=3, random_state=0,
            gamma=0.6, lambd=0.035, prediction_batch_size=2,
            sampling="with_replacement",
        ).fit(self.X, self.y)
        reference = independent_trajectory("kernel", self.X, self.y, 3, 3,
                                           sampling="with_replacement")
        self.assertTrue(any(len(np.unique(row["indices"])) < len(row["indices"]) for row in reference))
        self.assertEqual(model.n_updates_, 6)
        self.assertEqual(model.n_samples_seen_, 15)
        assert_allclose(model.dual_coef_.ravel(), reference[-1]["coefficients"], rtol=2e-13, atol=2e-14)
        self.assertAlmostEqual(model.intercept_[0], reference[-1]["intercept"], places=13)

    def test_full_batch_updates_match_independent_all_row_gradients(self):
        for kind, constructor in (("linear", LinearSGDDWD), ("kernel", KernelSGDDWD)):
            options = {} if kind == "linear" else {"gamma": 0.6, "lambd": 0.035,
                                                     "prediction_batch_size": 2}
            model = constructor(batch_size="full", max_epochs=4, random_state=17, **options).fit(self.X, self.y)
            reference = independent_trajectory(kind, self.X, self.y, "full", 4)
            coefficients = model.coef_.ravel() if kind == "linear" else model.dual_coef_.ravel()
            with self.subTest(estimator=constructor.__name__):
                self.assertEqual(model.n_updates_, 4)
                self.assertEqual(model.n_samples_seen_, 20)
                self.assertEqual(model.n_epochs_, 4)
                self.assertEqual(model.batch_size_, 5)
                self.assertEqual(model.sampling_, "full")
                assert_allclose(coefficients, reference[-1]["coefficients"], rtol=2e-13, atol=2e-14)
                self.assertAlmostEqual(model.intercept_[0], reference[-1]["intercept"], places=13)

    def test_full_batch_does_not_consume_sampling_rng_or_depend_on_seed(self):
        for sampling in ("shuffle", "with_replacement"):
            for constructor in (LinearSGDDWD, KernelSGDDWD):
                with self.subTest(estimator=constructor.__name__, sampling=sampling):
                    first = constructor(batch_size="full", sampling=sampling,
                                        max_epochs=4, random_state=17).fit(self.X, self.y)
                    second = constructor(batch_size="full", sampling=sampling,
                                         max_epochs=4, random_state=91).fit(self.X, self.y)
                    assert_array_equal(first.decision_function(self.X), second.decision_function(self.X))
                    self.assertEqual(first.sampling_, "full")
                    self.assertEqual(pickle.dumps(first.initial_state_["rng_state"]),
                                     pickle.dumps(first.final_state_["rng_state"]))

    def test_shuffle_kernel_trajectory_matches_independent_epoch_permutations(self):
        model = KernelSGDDWD(
            batch_size=3, sampling="shuffle", max_epochs=3, random_state=0,
            gamma=0.6, lambd=0.035, prediction_batch_size=2,
        ).fit(self.X, self.y)
        reference = independent_trajectory("kernel", self.X, self.y, 3, 3,
                                           sampling="shuffle")
        assert_allclose(model.dual_coef_.ravel(), reference[-1]["coefficients"], rtol=2e-13, atol=2e-14)
        self.assertAlmostEqual(model.intercept_[0], reference[-1]["intercept"], places=13)

    def test_shuffle_visits_every_row_once_in_each_epoch(self):
        sampled_batches = []
        row_ids = {tuple(row): index for index, row in enumerate(self.X)}
        original_scores = _estimators._scores
        def record_queries(kind, queries, coefficients, intercept, *args, **kwargs):
            sampled_batches.append(np.array([row_ids[tuple(row)] for row in queries]))
            return original_scores(kind, queries, coefficients, intercept, *args, **kwargs)

        model = KernelSGDDWD(batch_size=3, sampling="shuffle", max_epochs=3, random_state=0)
        with patch.object(_estimators, "_scores", side_effect=record_queries):
            model.fit(self.X, self.y)
        self.assertEqual([len(batch) for batch in sampled_batches], [3, 2] * 3)
        rng = np.random.default_rng(0)
        for epoch in range(3):
            observed = np.concatenate(sampled_batches[2 * epoch:2 * epoch + 2])
            assert_array_equal(np.sort(observed), np.arange(len(self.X)))
            assert_array_equal(observed, rng.permutation(len(self.X)))

    def test_clamped_shuffle_batch_uses_all_rows_once_per_update(self):
        for kind, constructor in (("linear", LinearSGDDWD), ("kernel", KernelSGDDWD)):
            options = {} if kind == "linear" else {"gamma": 0.6, "lambd": 0.035}
            model = constructor(batch_size=100, sampling="shuffle", max_epochs=3,
                                random_state=0, **options).fit(self.X, self.y)
            reference = independent_trajectory(kind, self.X, self.y, 100, 3,
                                               sampling="shuffle")
            coefficients = model.coef_.ravel() if kind == "linear" else model.dual_coef_.ravel()
            with self.subTest(estimator=constructor.__name__):
                self.assertEqual(model.batch_size_, len(self.X))
                self.assertEqual(model.sampling_, "shuffle")
                self.assertEqual(model.n_updates_, 3)
                self.assertEqual(model.n_samples_seen_, 15)
                assert_allclose(coefficients, reference[-1]["coefficients"], rtol=2e-13, atol=2e-14)
                self.assertAlmostEqual(model.intercept_[0], reference[-1]["intercept"], places=13)
                expected_rng = np.random.default_rng(0)
                for _ in range(3):
                    expected_rng.permutation(len(self.X))
                self.assertEqual(pickle.dumps(model.final_state_["rng_state"]),
                                 pickle.dumps(expected_rng.bit_generator.state))

    def test_explicit_replacement_batch_one_matches_independent_legacy_sampling(self):
        for kind, constructor in (("linear", LinearSGDDWD), ("kernel", KernelSGDDWD)):
            options = {} if kind == "linear" else {"gamma": 0.6, "lambd": 0.035}
            model = constructor(batch_size=1, sampling="with_replacement", max_epochs=3,
                                random_state=0, **options).fit(self.X, self.y)
            reference = independent_trajectory(kind, self.X, self.y, 1, 3,
                                               sampling="with_replacement")
            coefficients = model.coef_.ravel() if kind == "linear" else model.dual_coef_.ravel()
            with self.subTest(estimator=constructor.__name__):
                self.assertEqual(model.sampling_, "with_replacement")
                assert_allclose(coefficients, reference[-1]["coefficients"], rtol=2e-13, atol=2e-14)
                self.assertAlmostEqual(model.intercept_[0], reference[-1]["intercept"], places=13)

    def test_full_batch_kernel_scoring_never_allocates_full_gram(self):
        shapes = []
        original_kernel = _estimators.rbf_kernel
        def record_block(X, Z, gamma):
            shapes.append((len(X), len(Z)))
            return original_kernel(X, Z, gamma)

        model = KernelSGDDWD(
            batch_size="full", max_epochs=3, random_state=1,
            prediction_batch_size=2,
        )
        with patch.object(_estimators, "rbf_kernel", side_effect=record_block):
            model.fit(self.X, self.y)
        self.assertTrue(shapes)
        self.assertNotIn((len(self.X), len(self.X)), shapes)
        self.assertTrue(all(rows * columns <= 2 * len(self.X) for rows, columns in shapes))
        self.assertTrue(np.all(np.isfinite(model.decision_function(self.X))))


class BatchStoppingAndPipelineTests(unittest.TestCase):
    def test_validation_occurs_at_complete_epochs_for_truncated_batches(self):
        X, y = tiny_data()
        for constructor in (LinearSGDDWD, KernelSGDDWD):
            with self.subTest(estimator=constructor.__name__):
                model = constructor(
                    batch_size=3, max_epochs=5, validation_interval=2,
                    min_epochs=100, random_state=0,
                ).fit(X, y, validation_data=(X, y))
                checks = [entry for entry in model.history_ if entry["validation_loss"] is not None]
                self.assertEqual([entry["epoch"] for entry in checks], [2, 4])
                self.assertEqual([entry["iteration"] for entry in checks], [4, 8])
                self.assertEqual([entry["n_samples_seen"] for entry in checks], [10, 20])
                self.assertEqual(model.n_updates_, 10)
                self.assertEqual(model.n_samples_seen_, 25)

    def test_full_batch_patience_uses_twenty_epochs_not_twenty_observations(self):
        X, y = np.zeros((4, 1)), np.array([-1, -1, 1, 1])
        model = LinearSGDDWD(batch_size="full", max_epochs=40, random_state=1).fit(
            X, y, validation_data=(X, y)
        )
        checks = [entry for entry in model.history_ if entry["validation_loss"] is not None]
        self.assertEqual([entry["epoch"] for entry in checks], [5, 10, 15, 20])
        self.assertEqual([entry["iteration"] for entry in checks], [5, 10, 15, 20])
        self.assertEqual([entry["patience_misses"] for entry in checks], [0, 1, 2, 3])
        self.assertEqual(model.n_updates_, 20)
        self.assertEqual(model.n_samples_seen_, 80)
        self.assertEqual(model.n_epochs_, 20)
        self.assertEqual(model.selected_iteration_, 5)
        self.assertEqual(model.selected_n_samples_seen_, 20)
        self.assertEqual(model.stop_reason_, "validation_patience")

    def test_two_stage_effective_batch_size_uses_each_actual_fitting_set(self):
        X, y = selection_data()
        for requested in (50, "full"):
            with self.subTest(batch_size=requested):
                model = TwoStageDWDClassifier(
                    batch_size=requested, retention_fraction=0.4,
                    max_epochs=2, random_state=7,
                ).fit(X, y)
                linear, kernel = model.linear_estimator_, model.kernel_estimator_
                self.assertEqual(linear.batch_size_, 40)
                self.assertEqual(kernel.batch_size_, 16)
                self.assertEqual(linear.batch_size, requested)
                self.assertEqual(kernel.batch_size, requested)
                self.assertEqual(linear.n_samples_seen_, 80)
                self.assertEqual(kernel.n_samples_seen_, 32)
                self.assertEqual(linear.n_updates_, 2)
                self.assertEqual(kernel.n_updates_, 2)
                self.assertEqual(len(np.unique(model.selected_indices_)), 16)

    def test_two_stage_minibatch_kernel_equals_direct_subset_fit(self):
        X, y = selection_data()
        model = TwoStageDWDClassifier(
            batch_size=3, retention_fraction=0.4, max_epochs=3,
            gamma=0.6, lambd=0.035, random_state=7,
        ).fit(X, y)
        selected = model.selected_indices_
        direct = KernelSGDDWD(
            batch_size=3, max_epochs=3, gamma=0.6, lambd=0.035,
            random_state=model.stage_seeds_["kernel_sgd"],
        ).fit(X[selected], y[selected])
        assert_array_equal(model.kernel_estimator_.dual_coef_, direct.dual_coef_)
        assert_array_equal(model.kernel_estimator_.intercept_, direct.intercept_)
        assert_array_equal(model.decision_function(X), direct.decision_function(X))

    def test_single_class_selection_is_not_repaired_in_new_batch_modes(self):
        X, y = selection_data()
        for requested in (3, "full"):
            model = TwoStageDWDClassifier(
                batch_size=requested, retention_fraction=0.01,
                max_epochs=2, random_state=7,
            )
            with self.subTest(batch_size=requested):
                with self.assertRaises(SelectionError) as caught:
                    model.fit(X, y)
                self.assertEqual(caught.exception.status, "single_class_selection")
                self.assertEqual(len(model.selected_indices_), 1)
                self.assertEqual(model.linear_estimator_.model_status_, "trained")
                with self.assertRaises(NotFittedError):
                    model.predict(X)

    def test_interrupted_second_kernel_batch_preserves_first_batch_atomically(self):
        X, y = tiny_data()
        clock, score_calls = [0.0], [0]
        original_scores = _estimators._scores

        def interrupt_second_batch(kind, queries, coefficients, intercept, *args, **kwargs):
            score_calls[0] += 1
            if score_calls[0] == 2:
                # Do one observation's scoring work, then interrupt before
                # all loss gradients in the second batch are available.
                original_scores(kind, queries[:1], coefficients, intercept, *args, **kwargs)
                clock[0] = 2.0
                raise _estimators._Deadline
            return original_scores(kind, queries, coefficients, intercept, *args, **kwargs)

        model = KernelSGDDWD(
            batch_size=3, max_epochs=3, gamma=0.6, lambd=0.035,
            prediction_batch_size=2, random_state=0, max_seconds=1.0,
        )
        with patch.object(_estimators.time, "perf_counter", side_effect=lambda: clock[0]), \
             patch.object(_estimators, "_scores", side_effect=interrupt_second_batch):
            model.fit(X, y)
        first = independent_trajectory("kernel", X, y, 3, 1)[0]
        self.assertEqual(score_calls[0], 2)
        self.assertEqual(model.n_updates_, 1)
        self.assertEqual(model.n_samples_seen_, 3)
        self.assertAlmostEqual(model.n_epochs_, 3 / 5)
        self.assertEqual(model.final_state_["n_samples_seen"], 3)
        self.assertEqual(model.selected_n_samples_seen_, 3)
        self.assertEqual(model.stop_reason_, "time_cap_no_validation")
        self.assertEqual(model.model_status_, "trained")
        assert_allclose(model.dual_coef_.ravel(), first["coefficients"], rtol=2e-13, atol=2e-14)
        self.assertAlmostEqual(model.intercept_[0], first["intercept"], places=13)
        self.assertEqual(model.kernel_evaluation_count_, (3 + 1) * len(X))


if __name__ == "__main__":
    unittest.main()
