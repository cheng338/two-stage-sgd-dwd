"""Post-fit parent prediction controls reach each real kernel block statelessly."""
import copy
import pickle
from contextlib import contextmanager

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from two_stage_sgd_dwd import TwoStageDWDClassifier
from two_stage_sgd_dwd import _estimators, _prediction


def linear_matrix(X, Y):
    """A pickleable custom kernel with a positive semidefinite training Gram."""
    return X @ Y.T + 0.2


ROUTES = (
    ("rbf-direct", dict(kernel="rbf", prediction_backend="direct")),
    ("rbf-blas", dict(kernel="rbf", prediction_backend="blas")),
    ("rbf-zero", dict(kernel="rbf", gamma=0.0)),
    ("linear", dict(kernel="linear")),
    ("poly", dict(kernel="poly", kernel_kws={"degree": 2, "coef0": 0.2})),
    ("polynomial", dict(kernel="polynomial", kernel_kws={"degree": 2})),
    ("laplacian", dict(kernel="laplacian")),
    ("cosine", dict(kernel="cosine")),
    ("chi2", dict(kernel="chi2")),
    # These named kernels require a PSD check of the actual fitting matrix.
    # Constant sigmoid and identical additive-chi2 fitting rows satisfy it.
    ("sigmoid", dict(kernel="sigmoid", gamma=0.0, kernel_kws={"coef0": 0.5})),
    ("additive_chi2", dict(kernel="additive_chi2")),
    ("callable", dict(kernel=linear_matrix)),
)


def data(constant=False, n_features=3):
    x = np.linspace(0.1, 1.5, 12)
    X = np.column_stack((x, 0.2 + x * x / 4, 0.1 + x / 3))
    if constant:
        X[:] = (0.3, 0.6, 0.9)
    y = np.array(["lower"] * 6 + ["upper"] * 6)
    query = np.column_stack((np.linspace(0.13, 1.61, 11),
                             np.linspace(0.25, 0.76, 11),
                             np.linspace(0.16, 0.61, 11)))
    # Zero padding retains the original geometry while exercising a feature
    # count outside the small direct-execution regime of the BLAS backend.
    if n_features > 3:
        X = np.pad(X, ((0, 0), (0, n_features - 3)))
        query = np.pad(query, ((0, 0), (0, n_features - 3)))
    labels = np.array(["lower"] * 5 + ["upper"] * 6)
    return X, y, query, labels


def fitted(settings=None, n_features=3):
    settings = {} if settings is None else settings
    X, y, query, labels = data(
        settings.get("kernel") == "additive_chi2", n_features=n_features)
    controls = dict(retention_fraction=1.0, lambd=0.03, gamma=0.4,
                    max_epochs=3, stopping="fixed", random_state=29,
                    prediction_batch_size=16, optimizer_backend="python")
    controls.update(settings)
    return TwoStageDWDClassifier(**controls).fit(X, y), query, labels


def reference_child(model):
    """Keep the fitted state, but configure an independent public estimator."""
    child = copy.deepcopy(model.kernel_estimator_)
    # Legacy pickles can omit this constructor attribute. Restore it only on
    # the independent reference before sklearn set_params calls get_params.
    if not hasattr(child, "prediction_backend"):
        child.prediction_backend = "direct"
    child.set_params(prediction_batch_size=model.prediction_batch_size,
                     prediction_backend=getattr(model, "prediction_backend", "direct"))
    return child


def public_call(model, operation, query, labels):
    if operation == "score":
        return model.score(query, labels)
    return getattr(model, operation)(query)


@contextmanager
def observed_blocks(monkeypatch, model, before_block=None):
    """Observe actual numerical block boundaries, not the proposed helper API."""
    child = model.kernel_estimator_
    unchanged = pickle.dumps(child)
    sizes = []

    def observe(left):
        # Detect temporary child mutation too, even if restored after scoring.
        assert pickle.dumps(child) == unchanged
        sizes.append(len(left))
        if before_block is not None:
            before_block(len(sizes), left)

    with monkeypatch.context() as patch:
        if child._kernel_spec_.kernel == "rbf" and child.gamma_ > 0:
            # These fixtures have eleven query rows and at most twelve basis
            # rows. The three-feature BLAS fixture must execute directly; its
            # 33-feature counterpart must start each block with norm/dot work.
            # Observe the required route, so an accidental dispatch change
            # fails the block assertions rather than adapting this observer.
            if (getattr(model, "prediction_backend", "direct") == "blas"
                    and child.n_features_in_ == 33):
                original = _prediction.np.matmul

                def multiply(left, right, *args, **kwargs):
                    observe(left)
                    return original(left, right, *args, **kwargs)

                patch.setattr(_prediction.np, "matmul", multiply)
            else:
                original = _prediction.cdist

                def distances(left, right, *args, **kwargs):
                    observe(left)
                    return original(left, right, *args, **kwargs)

                patch.setattr(_prediction, "cdist", distances)

                def unexpected_multiply(*args, **kwargs):
                    raise AssertionError("Direct prediction block used norm/dot work")

                patch.setattr(_prediction.np, "matmul", unexpected_multiply)
        else:
            original = _estimators._kernel_values

            def kernel(left, right, *args, **kwargs):
                observe(left)
                return original(left, right, *args, **kwargs)

            patch.setattr(_estimators, "_kernel_values", kernel)
        yield sizes


@pytest.mark.parametrize("route,settings", ROUTES, ids=[row[0] for row in ROUTES])
@pytest.mark.parametrize("size", [1, 3, np.int64(4), 100])
@pytest.mark.parametrize("n_features", [3, 33], ids=["small-direct", "large-blas"])
def test_post_fit_parent_size_reaches_real_blocks_and_public_readouts(
        route, settings, size, n_features, monkeypatch):
    model, query, labels = fitted(settings, n_features=n_features)
    child_before = pickle.dumps(model.kernel_estimator_)
    model.set_params(prediction_batch_size=size)
    model_before = pickle.dumps(model)
    reference = reference_child(model)
    expected = {operation: public_call(reference, operation, query, labels)
                for operation in ("decision_function", "predict", "score")}
    expected_blocks = [min(int(size), len(query) - start)
                       for start in range(0, len(query), int(size))]

    for operation in expected:
        with observed_blocks(monkeypatch, model) as sizes:
            actual = public_call(model, operation, query, labels)
        assert sizes == expected_blocks, (route, operation, size, sizes)
        # Both sides use the same effective block size and backend. No claim
        # about bitwise agreement across different BLAS block sizes is made.
        assert_array_equal(actual, expected[operation])
        assert pickle.dumps(model) == model_before
        assert pickle.dumps(model.kernel_estimator_) == child_before
    assert model.kernel_estimator_.prediction_batch_size == 16


@pytest.mark.parametrize("invalid", [None, 0, -1, -256, 1.0, 1.5,
                                      True, False, np.bool_(True), np.nan,
                                      np.inf, "3"])
@pytest.mark.parametrize("n_features", [3, 33], ids=["small-direct", "large-blas"])
def test_invalid_parent_sizes_reject_without_dispatch_and_recover(
        invalid, n_features, monkeypatch):
    model, query, labels = fitted(n_features=n_features)
    child_before = pickle.dumps(model.kernel_estimator_)
    model.set_params(prediction_batch_size=invalid)
    model_before = pickle.dumps(model)

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid parent batch size reached numerical prediction")

    with monkeypatch.context() as patch:
        patch.setattr(_estimators, "rbf_prediction_scores", unexpected)
        patch.setattr(_estimators, "rbf_prediction_scores_blas", unexpected)
        patch.setattr(_estimators, "_scores", unexpected)
        for operation in ("decision_function", "predict", "score"):
            with pytest.raises(ValueError, match="prediction_batch_size"):
                public_call(model, operation, query, labels)
            assert pickle.dumps(model) == model_before
            assert pickle.dumps(model.kernel_estimator_) == child_before

    model.set_params(prediction_batch_size=1)
    expected = reference_child(model).decision_function(query)
    with observed_blocks(monkeypatch, model) as sizes:
        assert_array_equal(model.decision_function(query), expected)
    assert sizes == [1] * len(query)
    assert pickle.dumps(model.kernel_estimator_) == child_before


@pytest.mark.parametrize("backend", ["direct", "blas"])
@pytest.mark.parametrize("error_type", [RuntimeError, FloatingPointError,
                                        KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("n_features", [3, 33], ids=["small-direct", "large-blas"])
def test_failed_numerical_call_preserves_exact_error_and_fitted_state(
        backend, error_type, n_features, monkeypatch):
    model, query, labels = fitted(dict(prediction_backend=backend),
                                  n_features=n_features)
    model.set_params(prediction_batch_size=3)
    model_before = pickle.dumps(model)
    child_before = pickle.dumps(model.kernel_estimator_)
    expected = reference_child(model).decision_function(query)
    error = error_type("injected second prediction block failure")

    def fail_second(index, left):
        if index == 2:
            raise error

    with observed_blocks(monkeypatch, model, fail_second) as sizes:
        with pytest.raises(error_type) as caught:
            model.predict(query)
    assert caught.value is error
    assert sizes == [3, 3]
    assert pickle.dumps(model) == model_before
    assert pickle.dumps(model.kernel_estimator_) == child_before
    assert_array_equal(model.decision_function(query), expected)
    assert pickle.dumps(model) == model_before


@pytest.mark.parametrize("bad_query", [np.ones((2, 2)), np.empty((0, 3)),
                                       np.full((2, 3), np.nan), np.ones(3)])
def test_invalid_features_preserve_state_after_parent_batch_change(bad_query):
    model, query, labels = fitted()
    model.set_params(prediction_batch_size=1)
    before = pickle.dumps(model)
    with pytest.raises(ValueError):
        model.predict(bad_query)
    assert pickle.dumps(model) == before
    assert_array_equal(model.predict(query), reference_child(model).predict(query))


@pytest.mark.parametrize("operation", ["decision_function", "predict", "score"])
def test_unfitted_parent_precedes_invalid_size_and_input(operation):
    model = TwoStageDWDClassifier(prediction_batch_size=0)
    with pytest.raises(NotFittedError):
        public_call(model, operation, np.ones(3), np.array([1]))


def test_child_fittedness_and_invalid_child_controls_remain_guarded():
    model, query, labels = fitted()
    model.set_params(prediction_batch_size=1)
    child = model.kernel_estimator_
    child.set_params(prediction_batch_size=0)
    before = pickle.dumps(model)
    with pytest.raises(ValueError, match="prediction_batch_size"):
        model.decision_function(query)
    assert pickle.dumps(model) == before
    child.set_params(prediction_batch_size=16)
    child._fitted_ = False
    with pytest.raises(NotFittedError):
        model.predict(query)


@pytest.mark.parametrize("missing_backend", [False, True])
def test_pickle_roundtrip_preserves_parent_override_and_legacy_backend(
        missing_backend, monkeypatch):
    model, query, labels = fitted(dict(prediction_backend="direct"))
    model.set_params(prediction_batch_size=np.int64(3))
    if missing_backend:
        del model.prediction_backend
        del model.kernel_estimator_.prediction_backend
    expected = reference_child(model).decision_function(query)
    restored = pickle.loads(pickle.dumps(model))
    before = pickle.dumps(restored)
    with observed_blocks(monkeypatch, restored) as sizes:
        assert_array_equal(restored.decision_function(query), expected)
    assert sizes == [3, 3, 3, 2]
    assert pickle.dumps(restored) == before
    assert restored.kernel_estimator_.prediction_batch_size == 16
    assert hasattr(restored, "prediction_backend") is (not missing_backend)
    assert hasattr(restored.kernel_estimator_, "prediction_backend") is (not missing_backend)


def test_clone_and_refit_keep_configured_size_without_reusing_old_child(monkeypatch):
    model, query, labels = fitted(dict(prediction_backend="direct"))
    old_child = model.kernel_estimator_
    old_before = pickle.dumps(old_child)
    model.set_params(prediction_batch_size=np.int64(3))
    copied = clone(model)
    assert copied.prediction_batch_size == 3
    assert not hasattr(copied, "kernel_estimator_")
    X, y, _, _ = data()
    copied.fit(X, y)
    model.fit(X, y)
    assert model.kernel_estimator_ is not old_child
    assert copied.kernel_estimator_.prediction_batch_size == 3
    assert model.kernel_estimator_.prediction_batch_size == 3
    assert pickle.dumps(old_child) == old_before
    assert_array_equal(model.selected_ids_, copied.selected_ids_)
    assert_array_equal(model.kernel_estimator_.dual_coef_, copied.kernel_estimator_.dual_coef_)
    assert_array_equal(model.kernel_estimator_.intercept_, copied.kernel_estimator_.intercept_)
    for fitted_model in (model, copied):
        expected = reference_child(fitted_model).decision_function(query)
        with observed_blocks(monkeypatch, fitted_model) as sizes:
            assert_array_equal(fitted_model.decision_function(query), expected)
        assert sizes == [3, 3, 3, 2]
