"""Selection composition describes actual rows without changing their selection."""
from copy import deepcopy
import json

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from two_stage_sgd_dwd import SelectionError, TwoStageDWDClassifier
from two_stage_sgd_dwd._selection_reporting import selection_composition


def imbalanced_data(labels=("majority", "minority")):
    X = np.r_[-np.linspace(.05, 1.2, 12), np.linspace(.08, 1.5, 6)][:, None]
    return X, np.repeat(np.asarray(labels), [12, 6])


def assert_composition(report, training_y, selected_y):
    """Check denominators against observations rather than the reporting helper."""
    classes, training_counts = np.unique(training_y, return_counts=True)
    assert report["available"] is True
    assert report["training_size"] == len(training_y)
    assert report["selected_size"] == len(selected_y)
    assert report["selection_ratio_realized"] == len(selected_y) / len(training_y)
    assert len(report["classes"]) == len(classes)
    for row, label, count in zip(report["classes"], classes, training_counts):
        selected = int(np.count_nonzero(selected_y == label))
        assert row == {"label": label.item(), "training_count": int(count),
                       "training_fraction": count / len(training_y),
                       "selected_count": selected,
                       "selected_fraction": selected / len(selected_y) if len(selected_y) else None,
                       "class_retention_fraction": selected / count}
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("labels", [("zebra", "apricot"), (17, -91), (True, False)])
@pytest.mark.parametrize("fraction", [.75, 1.0])
def test_actual_imbalanced_selection_reports_original_labels_and_denominators(labels, fraction):
    X, y = imbalanced_data(labels)
    model = TwoStageDWDClassifier(retention_fraction=fraction, max_epochs=2,
                                  random_state=31, gamma=.4).fit(X, y)
    chosen = y[model.selected_indices_]
    assert len(np.unique(chosen)) == 2
    assert_composition(model.selection_composition_, y, chosen)
    assert_array_equal(model.training_class_counts_, np.unique(y, return_counts=True)[1])
    assert_array_equal(model.selected_class_counts_,
                       [np.count_nonzero(chosen == label) for label in model.classes_])
    if fraction < 1:
        assert all(row["selected_fraction"] != row["class_retention_fraction"]
                   for row in model.selection_composition_["classes"])
    else:
        assert all(row["selected_fraction"] == row["training_fraction"]
                   and row["class_retention_fraction"] == 1.
                   for row in model.selection_composition_["classes"])


@pytest.mark.parametrize("fraction,count,status", [
    (.01, 0, "empty_selection"), (.06, 1, "single_class_selection")])
def test_failed_selection_retains_counts_and_refit_clears_stale_receipts(fraction, count, status):
    X, y = imbalanced_data()
    model = TwoStageDWDClassifier(retention_fraction=1., max_epochs=2,
                                  random_state=31, gamma=.4).fit(X, y)
    old = deepcopy(model.selection_composition_)
    model.set_params(retention_fraction=fraction)
    with pytest.raises(SelectionError) as caught:
        model.fit(X, y)
    assert caught.value.status == status
    assert model.n_retained_ == count
    assert_composition(model.selection_composition_, y, y[model.selected_indices_])
    assert model.selection_composition_ != old
    assert not hasattr(model, "kernel_estimator_")
    # An invalid subsequent fit must not retain either successful or failed evidence.
    with pytest.raises(ValueError):
        model.fit(X, np.repeat("only", len(y)))
    assert not hasattr(model, "selection_composition_")
    assert not hasattr(model, "training_class_counts_")


@pytest.mark.parametrize("training,selected", [
    (None, [2, 1]), ([3, 2], None), ([3], [2, 1]), ([3, 2], [4, 1]),
    ([3, 0], [2, 0]), ([3, 2], [-1, 1]), ([3., 2.], [2, 1]),
    ([True, True], [1, 1]), ([[3], [2, 1]], [2, 1]),
    (np.array([2**63 - 1, 1], dtype=np.int64),
     np.array([2**63, 1], dtype=np.uint64)),
])
def test_missing_or_invalid_legacy_counts_remain_unavailable(training, selected):
    report = selection_composition(["a", "b"], training, selected)
    assert report["available"] is False
    assert "classes" not in report and "training_size" not in report
    json.dumps(report, allow_nan=False)


def test_receipt_mutation_cannot_change_count_evidence():
    labels = np.array(["a", "b"])
    training, selected = np.array([12, 6]), np.array([8, 5])
    report = selection_composition(labels, training, selected)
    report["classes"][0]["training_count"] = 999
    assert_array_equal(training, [12, 6])
    assert selection_composition(labels, training, selected)["classes"][0]["training_count"] == 12
