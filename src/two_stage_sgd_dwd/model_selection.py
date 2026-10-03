"""Explicit, training-only parameter searches for two-stage DWD.

Preprocessing belongs inside a sklearn Pipeline. The helpers never create a
held-out test set or run a search until the returned search object's ``fit`` is
called. Cross-validation scoring data must not select SGD checkpoints.
"""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.metrics import check_scoring
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from ._core import dwd_loss
from ._estimators import TrainingError, TwoStageDWDClassifier


def mnist_param_grid(prefix="", *, broad=False):
    """Return a two-stage RBF starting grid for scaled MNIST features.

    The lambda/gamma ranges and retention choices follow ensemble DWD's MNIST
    preset: 10%, 15%, 20%, 25%, and 30% retention. The resulting
    125 candidates all use ``kernel='rbf'``. ``broad=True`` uses nine values
    for each coefficient and the same five retention fractions, giving
    405 candidates. These are starting ranges, not a claim of optimal
    settings or adequate range coverage for a particular preprocessing choice.

    ``lambd`` multiplies the squared RKHS norm, as in the previous DWD packages.
    Its coarse values are ``2**e`` for e=-30,-22,-14,-6,0; the broad exponents
    are -30,-26,-22,-18,-14,-10,-6,-2,0. Gamma uses powers of ten at exponents
    -8,-6,-4,-2,0, or every integer from -8 through 0 for the broad grid.

    ``prefix`` is prepended literally, for example ``"dwd__"`` for a Pipeline
    step named ``dwd``. Each call returns an independent dictionary and lists.
    Boundary expansion, refinement, fitting, and CV are never automatic.
    """
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    if not isinstance(broad, bool):
        raise TypeError("broad must be a boolean")
    lambda_exponents = ((-30, -26, -22, -18, -14, -10, -6, -2, 0)
                        if broad else (-30, -22, -14, -6, 0))
    gamma_exponents = range(-8, 1) if broad else range(-8, 1, 2)
    return {
        prefix + "kernel": ["rbf"],
        prefix + "lambd": [2.0 ** exponent for exponent in lambda_exponents],
        prefix + "gamma": [10.0 ** exponent for exponent in gamma_exponents],
        prefix + "retention_fraction": [0.1, 0.15, 0.2, 0.25, 0.3],
    }


def _require_trained(estimator):
    """Reject fallbacks and truncated fits; fixed epoch caps remain eligible."""
    if getattr(estimator, "model_status_", None) == "untrained_fallback":
        raise TrainingError(
            "An untrained_fallback model is ineligible for parameter selection",
            status="untrained_fallback",
        )
    if not getattr(estimator, "cv_eligible_", True):
        raise TrainingError(
            "The model is ineligible for parameter selection under its stopping diagnostics",
            status="ineligible_cv_fit",
        )
    if isinstance(estimator, Pipeline):
        _require_trained(estimator.steps[-1][1])
    for name in ("linear_estimator_", "kernel_estimator_"):
        stage = getattr(estimator, name, None)
        if stage is not None:
            _require_trained(stage)


def _finite_score(value):
    score = np.asarray(value)
    if (score.ndim != 0 or np.iscomplexobj(score)
            or not np.issubdtype(score.dtype, np.number)
            or not np.isfinite(score)):
        raise ValueError("scoring must return one finite real number")
    return float(score)


def neg_dwd_loss(estimator, X, y):
    """Score by negative mean normalized DWD loss, without regularization.

    This sklearn scorer accepts a fitted classifier or Pipeline and uses
    ``classes_[1]`` as the positive decision-function class. Evaluation labels
    must belong to its two fitted classes. Higher values are better. An
    untrained fallback in either stage is rejected before evaluating scores.
    """
    _require_trained(estimator)
    classes = np.asarray(estimator.classes_)
    if classes.ndim != 1 or len(classes) != 2:
        raise ValueError("neg_dwd_loss requires exactly two fitted classes")
    labels = np.asarray(y)
    if (labels.ndim != 1 or len(labels) == 0
            or not np.isin(labels, classes).all()):
        raise ValueError("y must be a nonempty vector of fitted class labels")
    scores = np.asarray(estimator.decision_function(X))
    if (scores.ndim != 1 or len(scores) != len(labels)
            or np.iscomplexobj(scores)
            or not np.issubdtype(scores.dtype, np.number)
            or not np.isfinite(scores).all()):
        raise ValueError("decision_function must return one finite real score per label")
    encoded = np.where(labels == classes[1], 1.0, -1.0)
    return _finite_score(-np.mean(dwd_loss(encoded * scores)))


class _TrainedScorer:
    """Picklable guard shared by accuracy and caller-supplied scalar scorers."""

    def __init__(self, scorer):
        self.scorer = scorer

    def __call__(self, estimator, X, y, **kwargs):
        _require_trained(estimator)
        return _finite_score(self.scorer(estimator, X, y, **kwargs))


class _TrainingOnlyGridSearchCV(GridSearchCV):
    """GridSearchCV with a guard against shared monitoring-set leakage."""

    def fit(self, X, y=None, **params):
        monitoring = [key for key in params
                      if key.split("__")[-1] == "validation_data"]
        if monitoring:
            raise ValueError(
                "make_search does not accept validation_data, including Pipeline "
                "step__validation_data. A shared monitoring set can leak across "
                "CV folds; use fixed epochs, or implement an explicit inner split "
                "within each training fold."
            )
        result = super().fit(X, y, **params)
        if hasattr(self, "best_estimator_"):
            _require_trained(self.best_estimator_)
        return result


def make_search(estimator=None, param_grid=None, *, cv=5, scoring="accuracy",
                random_state=0, n_jobs=1):
    """Build a fail-fast sklearn grid search; call ``fit`` to run it.

    ``param_grid`` must be supplied explicitly: appropriate ranges depend on
    the data, preprocessing, and scientific question. :func:`mnist_param_grid`
    is an opt-in scaled-MNIST preset, never an automatic default. The default
    estimator is ``TwoStageDWDClassifier(random_state=random_state)``. Supply
    an explicitly prefixed grid for a Pipeline, or a suitable grid for another
    estimator.
    A supplied estimator is preserved; its own random seed is not overwritten.

    Integer ``cv`` creates a shuffled, seeded ``StratifiedKFold``. Explicit
    splitters or fold iterables pass through unchanged. ``scoring`` is a single
    sklearn scorer name, scalar scorer callable, or None for estimator.score;
    use :func:`neg_dwd_loss` for normalized DWD loss instead of accuracy.
    Every scorer is guarded against untrained fallbacks, explicitly ineligible
    fits (including wall-time truncation), and nonfinite scores. A trained model
    that completes its epoch budget can receive a score without an optimality
    certificate. Reaching an epoch cap does not automatically increase it.
    Errors propagate (``error_score='raise'``); selection failures are not
    silently excluded. The final refit is checked for untrained fallbacks too.

    The returned GridSearchCV subclass provides standard ``cv_results_`` and
    ``best_*`` attributes. Its fit rejects direct or Pipeline-prefixed
    ``validation_data``. Use fixed epochs for this helper. Any early stopping
    requires a separate workflow that creates monitoring data inside each
    training fold. Reserve held-out data for evaluation after model selection.
    No adaptive expansion or held-out evaluation is performed here. Ordinary
    fixed-grid CV refits its selected candidate without an adaptive-boundary
    qualification; use the separate adaptive example for that review policy.
    """
    if param_grid is None:
        raise ValueError(
            "param_grid must be supplied explicitly; choose ranges for your data "
            "and preprocessing. mnist_param_grid is an opt-in MNIST preset."
        )
    if isinstance(cv, (bool, np.bool_)):
        raise ValueError("cv must be an integer >= 2 or an explicit CV splitter")
    if isinstance(cv, numbers.Integral):
        if cv < 2:
            raise ValueError("cv must be an integer >= 2")
        cv = StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=random_state)
    if estimator is None:
        estimator = TwoStageDWDClassifier(random_state=random_state)
    if scoring is not None and not isinstance(scoring, str) and not callable(scoring):
        raise TypeError("make_search accepts a single scorer name or callable")
    scorer = _TrainedScorer(check_scoring(estimator, scoring=scoring))
    return _TrainingOnlyGridSearchCV(
        estimator, param_grid, scoring=scorer, cv=cv, n_jobs=n_jobs,
        pre_dispatch="n_jobs", error_score="raise",
    )
