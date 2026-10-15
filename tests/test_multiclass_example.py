"""External one-versus-one composition preserves each binary SGD fit."""

import copy
import importlib.util
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.multiclass import OneVsOneClassifier
from threadpoolctl import threadpool_limits

from two_stage_sgd_dwd import SelectionError, TwoStageDWDClassifier

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None


_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "multiclass_example.py"
_SPEC = importlib.util.spec_from_file_location("multiclass_example", _EXAMPLE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
pair_diagnostics = _MODULE.pair_diagnostics


def multiclass_problem(n_classes, label_kind):
    """Shuffled rows and unsorted labels exercise the pair-to-label mapping."""
    rng = np.random.default_rng(19)
    centers = np.array([[-2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [2.0, -2.0]])
    labels = (
        np.array(["zebra", "apricot", "middle", "birch"])
        if label_kind == "strings"
        else np.array([17, -91, 208, 4])
    )
    X = np.vstack([
        center + rng.normal(scale=1.2, size=(24, 2))
        for center in centers[:n_classes]
    ])
    y = np.repeat(labels[:n_classes], 24)
    order = rng.permutation(len(y))
    return X[order], y[order]


def binary_prototype(**settings):
    options = dict(
        max_epochs=3,
        random_state=31,
        stopping="fixed",
        sampling="with_replacement",
        gamma=0.4,
        prediction_batch_size=7,
    )
    options.update(settings)
    return TwoStageDWDClassifier(**options)


@pytest.mark.parametrize("backend", ["python", "auto"])
@pytest.mark.parametrize(
    "n_classes,label_kind,batch_size,retention",
    [
        (3, "strings", 1, 1.0),
        (4, "integers", 7, 1.0),
        (4, "integers", 1, 0.3),
        (3, "strings", 7, 0.3),
    ],
)
def test_ovo_matches_manual_binary_fits_and_maps_pair_diagnostics(
    backend, n_classes, label_kind, batch_size, retention
):
    X, y = multiclass_problem(n_classes, label_kind)
    before_X, before_y = X.copy(), y.copy()
    prototype = binary_prototype(
        optimizer_backend=backend, batch_size=batch_size, retention_fraction=retention
    )
    with threadpool_limits(limits=1):
        model = OneVsOneClassifier(prototype, n_jobs=1).fit(X, y)
        reports = pair_diagnostics(model)
        pairs = list(combinations(np.unique(y), 2))
        assert len(model.estimators_) == len(pairs) == len(reports)
        assert model.pairwise_indices_ is None
        assert not hasattr(prototype, "classes_")
        expected_backend = (
            "native" if backend == "auto" and batch_size == 1 and _sgd_accel is not None
            else "python"
        )

        for fitted, report, (first, second) in zip(model.estimators_, reports, pairs):
            # sklearn supplies only the pair's rows, in their original order,
            # and maps its sorted labels to zero and one for the binary fit.
            mask = (y == first) | (y == second)
            pair_X = X[mask]
            pair_y = (y[mask] == second).astype(int)
            manual = clone(prototype).fit(pair_X, pair_y)
            assert_array_equal(fitted.classes_, [0, 1])
            assert fitted.stage_seeds_ == manual.stage_seeds_
            for name in (
                "selected_indices_", "selected_ids_", "selected_class_counts_",
                "selection_scores_",
            ):
                assert_array_equal(getattr(fitted, name), getattr(manual, name))
            for name in ("coef_", "intercept_"):
                assert_array_equal(
                    getattr(fitted.linear_estimator_, name),
                    getattr(manual.linear_estimator_, name),
                )
            for name in ("dual_coef_", "intercept_", "basis_"):
                assert_array_equal(
                    getattr(fitted.kernel_estimator_, name),
                    getattr(manual.kernel_estimator_, name),
                )
            query = X[[0, 9, 21, 35]] + np.array([0.03, -0.02])
            assert_array_equal(fitted.decision_function(query), manual.decision_function(query))
            assert_array_equal(fitted.predict(query), manual.predict(query))
            assert fitted.n_retained_ == int(np.floor(retention * len(pair_y)))
            assert np.all(fitted.selected_class_counts_ > 0)

            assert report["classes"] == (first, second)
            assert report["n_samples_fit"] == len(pair_y)
            assert report["n_retained"] == fitted.n_retained_
            selected_labels = y[mask][fitted.selected_indices_]
            assert report["selected_class_counts"] == [
                int(np.count_nonzero(selected_labels == first)),
                int(np.count_nonzero(selected_labels == second)),
            ]
            assert report["model_status"] == fitted.model_status_ == "trained"
            assert report["cv_eligible"] is True
            assert report["stage_seeds"] == fitted.stage_seeds_
            assert report["stage_diagnostics"] == fitted.stage_diagnostics_
            for stage in ("linear", "kernel"):
                diagnostic = report["stage_diagnostics"][stage]
                assert diagnostic["optimizer_backend"] == expected_backend
                assert diagnostic["termination_reason"] == "epoch_cap"
                assert diagnostic["cap_reached"]
                assert not diagnostic["optimality_checked"]
                assert not diagnostic["converged"]

        scores = model.decision_function(X)
        predicted = model.predict(X)
    assert scores.shape == (len(X), n_classes)
    assert np.isfinite(scores).all()
    assert predicted.shape == y.shape
    assert_array_equal(predicted, model.classes_[np.argmax(scores, axis=1)])
    assert_array_equal(X, before_X)
    assert_array_equal(y, before_y)


def test_report_mutation_cannot_change_fitted_pair_diagnostics():
    X, y = multiclass_problem(3, "strings")
    with threadpool_limits(limits=1):
        model = OneVsOneClassifier(
            binary_prototype(retention_fraction=1.0, optimizer_backend="python"),
            n_jobs=1,
        ).fit(X, y)
    reports = pair_diagnostics(model)
    before = copy.deepcopy(reports)
    reports[0]["selected_class_counts"][0] = -123
    reports[0]["stage_seeds"]["linear_sgd"] = -123
    reports[0]["stage_diagnostics"]["linear"]["n_iter"] = -123
    assert pair_diagnostics(model) == before
    fitted = model.estimators_[0]
    assert fitted.selected_class_counts_[0] >= 0
    assert fitted.stage_seeds_["linear_sgd"] >= 0
    assert fitted.stage_diagnostics_["linear"]["n_iter"] > 0


@pytest.mark.parametrize("backend", ["python", "auto"])
@pytest.mark.parametrize("fraction,count,status", [
    (.001, 0, "empty_selection"), (.025, 1, "single_class_selection"),
])
def test_insufficient_selection_propagates_failure_without_retry(backend, fraction, count, status):
    X, y = multiclass_problem(3, "strings")
    attempts = []
    original_fit = TwoStageDWDClassifier.fit

    def recorded_fit(estimator, pair_X, pair_y, **kwargs):
        attempts.append((estimator, pair_X.copy(), pair_y.copy()))
        return original_fit(estimator, pair_X, pair_y, **kwargs)

    model = OneVsOneClassifier(
        binary_prototype(retention_fraction=fraction, optimizer_backend=backend),
        n_jobs=1,
    )
    with threadpool_limits(limits=1), patch.object(
        TwoStageDWDClassifier, "fit", recorded_fit
    ), pytest.raises(SelectionError) as caught:
        model.fit(X, y)
    assert caught.value.status == status
    assert len(attempts) == 1
    failed, pair_X, pair_y = attempts[0]
    first, second = np.unique(y)[:2]
    mask = (y == first) | (y == second)
    assert_array_equal(pair_X, X[mask])
    assert_array_equal(pair_y, (y[mask] == second).astype(int))
    assert failed.n_retained_ == count
    assert failed.selected_class_counts_.sum() == count
    assert failed.model_status_ == status
    assert failed.stop_reason_ == "selection_failure"
    assert not failed.cv_eligible_
    assert not hasattr(failed, "kernel_estimator_")


def test_diagnostics_require_a_fitted_multiclass_wrapper():
    with pytest.raises(NotFittedError):
        pair_diagnostics(OneVsOneClassifier(binary_prototype()))
