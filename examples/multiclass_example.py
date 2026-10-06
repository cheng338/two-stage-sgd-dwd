"""External one-vs-one composition; not a multiclass performance benchmark."""
from copy import deepcopy
from itertools import combinations

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from two_stage_sgd_dwd import TwoStageDWDClassifier


def pair_diagnostics(model):
    """Snapshot successful OvO pair fits in their original class-label order.

    ``model`` must wrap TwoStageDWDClassifier directly, as in main(). The
    binary estimators see labels 0 and 1 for the first and second class of
    each pair. Their selected indices and IDs refer to that pair's input
    rows, not the full multiclass training data. Call only after fit succeeds.
    """
    check_is_fitted(model, ["classes_", "estimators_"])
    return [
        {
            "classes": pair,
            "n_samples_fit": int(estimator.n_samples_fit_),
            "n_retained": int(estimator.n_retained_),
            "selected_class_counts": estimator.selected_class_counts_.tolist(),
            "model_status": estimator.model_status_,
            "cv_eligible": bool(estimator.cv_eligible_),
            "stage_diagnostics": deepcopy(estimator.stage_diagnostics_),
            "stage_seeds": dict(estimator.stage_seeds_),
        }
        for pair, estimator in zip(
            combinations(model.classes_, 2), model.estimators_, strict=True
        )
    ]


def main():
    X, y = make_blobs(n_samples=120, centers=3, n_features=4,
                      cluster_std=1.5, random_state=17)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=19
    )
    model = make_pipeline(
        StandardScaler(),
        OneVsOneClassifier(
            TwoStageDWDClassifier(
                # Retain all rows to keep this tiny demonstration reliable;
                # this is not a retention recommendation or a new default.
                retention_fraction=1.0, lambd=5e-4, gamma="scale",
                max_epochs=20, stopping="fixed", optimizer_backend="auto",
                random_state=29,
            ),
            n_jobs=1,
        ),
    )
    with threadpool_limits(1):
        # The pipeline fits scaling only on the training data.
        model.fit(Xtrain, ytrain)
        print({
            "example": "synthetic_multiclass_not_mnist",
            "accuracy": model.score(Xtest, ytest),
            "pairs": pair_diagnostics(model.named_steps["onevsoneclassifier"]),
        })


if __name__ == "__main__":
    main()
