"""Small CV and scorer contracts; no benchmark datasets or search studies."""

import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from two_stage_sgd_dwd import (
    KernelSGDDWD, LinearSGDDWD, NumericalFailure, SelectionError,
    TrainingError, TwoStageDWDClassifier,
)
from two_stage_sgd_dwd.model_selection import make_search, mnist_param_grid, neg_dwd_loss


def small_data():
    magnitude = np.linspace(0.2, 1.8, 12)
    positive = np.column_stack((magnitude, 0.12 * np.sin(np.arange(12))))
    return np.vstack((-positive, positive)), np.array(["left"] * 12 + ["right"] * 12)


class StatusClassifier(ClassifierMixin, BaseEstimator):
    """Exercise search failure policy without running an optimizer."""

    def __init__(self, fallback_at=0, failure=None):
        self.fallback_at = fallback_at
        self.failure = failure

    def fit(self, X, y):
        if self.failure == "numerical":
            raise NumericalFailure("deliberate numerical failure")
        if self.failure == "selection":
            raise SelectionError("deliberate selection failure")
        self.classes_ = np.unique(y)
        self.model_status_ = ("untrained_fallback" if len(X) >= self.fallback_at
                              else "trained")
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))


class GridTests(unittest.TestCase):
    def test_default_and_broad_grids_tune_two_stage_penalty_gamma_and_retention(self):
        for broad, count, penalty_exponents, gamma_exponents in [
            (False, 125, [-30, -22, -14, -6, 0], [-8, -6, -4, -2, 0]),
            (True, 405, [-30, -26, -22, -18, -14, -10, -6, -2, 0],
             [-8, -7, -6, -5, -4, -3, -2, -1, 0]),
        ]:
            with self.subTest(broad=broad):
                grid = mnist_param_grid(broad=broad)
                self.assertEqual(len(ParameterGrid(grid)), count)
                self.assertEqual(grid["lambd"], [2.0 ** e for e in penalty_exponents])
                self.assertEqual(grid["gamma"], [10.0 ** e for e in gamma_exponents])
                self.assertEqual(grid["kernel"], ["rbf"])
                self.assertEqual(grid["retention_fraction"], [0.1, 0.15, 0.2, 0.25, 0.3])
                self.assertNotIn("penalty_lambda", grid)

    def test_broad_grid_preserves_coarse_endpoints_and_retention_choices(self):
        coarse, broad = mnist_param_grid(), mnist_param_grid(broad=True)
        for key in ("lambd", "gamma"):
            self.assertEqual(broad[key][::2], coarse[key])
            self.assertEqual(broad[key][0], coarse[key][0])
            self.assertEqual(broad[key][-1], coarse[key][-1])
        self.assertEqual(broad["retention_fraction"], coarse["retention_fraction"])

    def test_prefix_and_fresh_mutable_containers(self):
        expected = mnist_param_grid()
        prefixed = mnist_param_grid("dwd__")
        self.assertEqual(prefixed, {"dwd__" + k: v for k, v in expected.items()})
        for value in prefixed.values():
            value.append("changed")
        prefixed["extra"] = [1]
        self.assertEqual(mnist_param_grid(), expected)
        self.assertEqual(mnist_param_grid("literal")["literalkernel"], ["rbf"])

    def test_grid_argument_validation_matches_ensemble_helper(self):
        for prefix in [None, 1, [], False]:
            with self.subTest(prefix=prefix), self.assertRaises(TypeError):
                mnist_param_grid(prefix)
        for broad in [None, 0, 1, "yes", np.bool_(True)]:
            with self.subTest(broad=broad), self.assertRaises(TypeError):
                mnist_param_grid(broad=broad)

    def test_preset_uses_only_public_two_stage_parameters(self):
        for broad in [False, True]:
            with self.subTest(broad=broad):
                model = TwoStageDWDClassifier()
                grid = mnist_param_grid(broad=broad)
                self.assertTrue(set(grid).issubset(model.get_params()))
                self.assertIs(model.set_params(**ParameterGrid(grid)[0]), model)
                self.assertEqual(model.lambd, 2.0 ** -30)


class LossScorerTests(unittest.TestCase):
    def fixed_model(self, classes, scores):
        return SimpleNamespace(classes_=np.asarray(classes), model_status_="trained",
                               decision_function=lambda X: np.asarray(scores))

    def test_class_order_mapping_and_normalized_loss_are_independent_of_penalty(self):
        # classes_[1], even when lexically first, corresponds to positive scores.
        model = self.fixed_model(["zebra", "ant"], [-2.0, 0.25, -0.5, 1.0])
        model.lambd = 1e100
        actual = neg_dwd_loss(model, np.zeros((4, 1)), ["zebra", "ant", "ant", "ant"])
        expected = -np.mean([0.125, 0.75, 1.5, 0.25])
        self.assertAlmostEqual(actual, expected)

    def test_large_integer_and_boolean_labels(self):
        for classes in [[-(2**50) - 3, 2**50 + 17], [False, True]]:
            with self.subTest(classes=classes):
                model = self.fixed_model(classes, [-1.0, 1.0])
                self.assertEqual(neg_dwd_loss(model, [[0], [1]], classes), -0.25)

    def test_scorer_validates_labels_shapes_and_finite_decisions(self):
        X = np.zeros((2, 1))
        valid_model = self.fixed_model(["left", "right"], [0.0, 1.0])
        for y in [[], ["unknown", "right"], [["left"], ["right"]], ["left"]]:
            with self.subTest(labels=y), self.assertRaises(ValueError):
                neg_dwd_loss(valid_model, X, y)
        for scores in [[np.nan, 1], [np.inf, 0], [[0], [1]], [1 + 1j, 0], ["x", "y"]]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                neg_dwd_loss(self.fixed_model(["left", "right"], scores), X, ["left", "right"])
        for classes in [["left"], ["a", "b", "c"], [["left", "right"]]]:
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                neg_dwd_loss(self.fixed_model(classes, [0, 1]), X, ["left", "right"])

    def test_pipeline_loss_matches_manual_decisions_without_regularizer(self):
        X, y = small_data()
        model = Pipeline([
            ("scale", StandardScaler()),
            ("dwd", KernelSGDDWD(lambd=0.2, max_epochs=3, random_state=9)),
        ]).fit(X, y)
        margin = np.where(y == model.classes_[1], 1, -1) * model.decision_function(X)
        expected = np.array([1 - u if u <= 0.5 else 0.25 / u for u in margin])
        self.assertAlmostEqual(neg_dwd_loss(model, X, y), -expected.mean())
        regularized = model[-1].objective(model[0].transform(X), y)
        self.assertGreater(regularized, -neg_dwd_loss(model, X, y))

    def test_fallback_guard_covers_pipeline_and_both_two_stage_estimators(self):
        good = self.fixed_model(["left", "right"], [0.0, 1.0])
        bad = SimpleNamespace(model_status_="untrained_fallback")
        variants = [bad, Pipeline([("dwd", bad)]),
                    SimpleNamespace(model_status_="trained", linear_estimator_=bad),
                    SimpleNamespace(model_status_="trained", kernel_estimator_=bad),
                    Pipeline([("dwd", SimpleNamespace(
                        model_status_="trained", linear_estimator_=good,
                        kernel_estimator_=bad))])]
        for model in variants:
            with self.subTest(model=model), self.assertRaises(TrainingError) as caught:
                neg_dwd_loss(model, [[0], [1]], ["left", "right"])
            self.assertEqual(caught.exception.status, "untrained_fallback")


class SearchTests(unittest.TestCase):
    def test_search_requires_explicit_grid_and_never_selects_mnist_preset(self):
        with patch("two_stage_sgd_dwd.model_selection.mnist_param_grid") as preset:
            with self.assertRaisesRegex(ValueError, "param_grid must be supplied explicitly"):
                make_search()
            with self.assertRaisesRegex(ValueError, "param_grid must be supplied explicitly"):
                make_search(LinearSGDDWD())
            preset.assert_not_called()

    def test_defaults_seed_cv_and_estimator_without_starting_fit(self):
        grid = {"lambd": [0.05], "gamma": [0.4], "retention_fraction": [1.0]}
        search = make_search(param_grid=grid)
        self.assertIsInstance(search, GridSearchCV)
        self.assertIsInstance(search.estimator, TwoStageDWDClassifier)
        self.assertEqual(search.estimator.random_state, 0)
        self.assertEqual(search.estimator.retention_fraction, 0.3)
        self.assertIs(search.param_grid, grid)
        self.assertEqual(search.cv.n_splits, 5)
        self.assertTrue(search.cv.shuffle)
        self.assertEqual(search.cv.random_state, 0)
        self.assertEqual(search.error_score, "raise")
        self.assertEqual(search.n_jobs, 1)
        self.assertFalse(hasattr(search, "cv_results_"))
        cloned = clone(search)
        self.assertEqual(cloned.param_grid, search.param_grid)
        self.assertIsNot(cloned.estimator, search.estimator)

    def test_explicit_splitter_and_estimator_are_preserved(self):
        X, y = small_data()
        model = LinearSGDDWD(random_state=92)
        splitter = StratifiedKFold(2, shuffle=True, random_state=17)
        for cv in [splitter, list(splitter.split(X, y))]:
            with self.subTest(cv=type(cv).__name__):
                search = make_search(model, {}, cv=cv, random_state=7)
                self.assertIs(search.cv, cv)
                self.assertIs(search.estimator, model)
                self.assertEqual(model.random_state, 92)

    def test_integer_cv_reproduces_splits_and_rejects_invalid_values(self):
        X, y = small_data()
        first = make_search(param_grid={}, cv=3, random_state=17).cv.split(X, y)
        second = make_search(param_grid={}, cv=3, random_state=17).cv.split(X, y)
        for (a, b), (c, d) in zip(first, second):
            assert_array_equal(a, c)
            assert_array_equal(b, d)
        for cv in [True, False, np.bool_(True), 0, 1, -1]:
            with self.subTest(cv=cv), self.assertRaises(ValueError):
                make_search(param_grid={}, cv=cv)

    def test_pipeline_cv_is_reproducible_and_refits_on_all_training_rows(self):
        X, y = small_data()
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("dwd", TwoStageDWDClassifier(
                retention_fraction=1.0, max_epochs=2, random_state=41)),
        ])
        grid = {"dwd__lambd": [0.01, 0.1], "dwd__gamma": [0.4], "dwd__kernel": ["rbf"]}
        first = make_search(estimator, grid, cv=2, random_state=11).fit(X, y)
        second = make_search(clone(estimator), grid, cv=2, random_state=11).fit(X, y)
        self.assertEqual(first.best_params_, second.best_params_)
        assert_array_equal(first.cv_results_["mean_test_score"], second.cv_results_["mean_test_score"])
        assert_array_equal(first.predict(X), second.predict(X))
        self.assertTrue(np.isfinite(first.best_score_))
        self.assertEqual(len(first.cv_results_["params"]), 2)
        assert_allclose(first.best_estimator_[0].mean_, X.mean(axis=0))
        self.assertEqual(len(first.best_estimator_[-1].kernel_estimator_.basis_), len(X))
        self.assertFalse(hasattr(estimator[-1], "model_status_"))
        restored = pickle.loads(pickle.dumps(first))
        assert_array_equal(restored.predict(X), first.predict(X))

    def test_neg_dwd_loss_works_as_grid_search_callable(self):
        X, y = small_data()
        search = make_search(LinearSGDDWD(random_state=7), {"max_epochs": [1, 2]},
                             cv=2, scoring=neg_dwd_loss).fit(X, y)
        self.assertLess(search.best_score_, 0)
        self.assertAlmostEqual(search.score(X, y), neg_dwd_loss(search.best_estimator_, X, y))

    def test_monitoring_data_is_rejected_before_cv_or_refit(self):
        X, y = small_data()
        search = make_search(LinearSGDDWD(), {}, cv=2)
        for key in ["validation_data", "dwd__validation_data", "outer__dwd__validation_data"]:
            with self.subTest(key=key), patch.object(GridSearchCV, "fit") as fit:
                with self.assertRaisesRegex(ValueError, "inner split"):
                    search.fit(X, y, **{key: (X, y)})
                fit.assert_not_called()

    def test_accuracy_and_custom_scoring_reject_untrained_candidates(self):
        X, y = small_data()
        for scoring in ["accuracy", None, lambda estimator, X, y: 1.0]:
            for model in [StatusClassifier(), Pipeline([("status", StatusClassifier())])]:
                with self.subTest(scoring=scoring, model=type(model).__name__):
                    with self.assertRaises(TrainingError) as caught:
                        make_search(model, {}, cv=2, scoring=scoring).fit(X, y)
                    self.assertEqual(caught.exception.status, "untrained_fallback")

    def test_final_refit_fallback_is_rejected_even_if_cv_fits_were_trained(self):
        X, y = small_data()
        search = make_search(StatusClassifier(fallback_at=len(X)), {}, cv=2)
        with self.assertRaises(TrainingError) as caught:
            search.fit(X, y)
        self.assertEqual(caught.exception.status, "untrained_fallback")

    def test_numerical_and_selection_errors_are_not_suppressed(self):
        X, y = small_data()
        for failure, error in [("numerical", NumericalFailure), ("selection", SelectionError)]:
            with self.subTest(failure=failure), self.assertRaises(error):
                make_search(StatusClassifier(failure=failure), {}, cv=2).fit(X, y)

    def test_nonfinite_or_nonscalar_custom_scores_fail_fast(self):
        X, y = small_data()
        for value in [np.nan, np.inf, [0.5], 1 + 2j, "bad"]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite real"):
                make_search(StatusClassifier(fallback_at=100), {}, cv=2,
                            scoring=lambda estimator, X, y: value).fit(X, y)
        with self.assertRaises(TypeError):
            make_search(param_grid={}, scoring=["accuracy", "balanced_accuracy"])


if __name__ == "__main__":
    unittest.main()
