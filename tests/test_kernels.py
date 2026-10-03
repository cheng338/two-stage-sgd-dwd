"""Kernel conventions, domain checks, and preparation contracts."""

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from two_stage_sgd_dwd._kernels import resolve_kernel


def linear_with_offset(X, Y, offset=0.0, gamma=1.0):
    return gamma * (X @ Y.T) + np.asarray(offset).reshape(-1)[0]


@pytest.mark.parametrize("name", ["linear", "rbf", "poly", "polynomial", "laplacian", "cosine", "chi2"])
def test_named_kernel_formulas_and_analytic_diagonal(name):
    X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.5]])
    Y = np.array([[0.5, 1.0], [1.0, 0.0]])
    gamma = 0.3
    kws = {"degree": 2, "coef0": 0.4} if name in ("poly", "polynomial") else None
    spec = resolve_kernel(X, name, gamma, kws)
    expected = np.empty((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if name == "linear":
                expected[i, j] = x @ y
            elif name == "rbf":
                expected[i, j] = np.exp(-gamma * np.sum((x - y) ** 2))
            elif name in ("poly", "polynomial"):
                expected[i, j] = (gamma * (x @ y) + 0.4) ** 2
            elif name == "laplacian":
                expected[i, j] = np.exp(-gamma * np.sum(np.abs(x - y)))
            elif name == "cosine":
                denominator = np.linalg.norm(x) * np.linalg.norm(y)
                expected[i, j] = x @ y / denominator if denominator else 0.0
            else:
                positive = x + y > 0
                distance = np.sum((x[positive] - y[positive]) ** 2 / (x[positive] + y[positive]))
                expected[i, j] = np.exp(-gamma * distance)
    assert_allclose(spec.evaluate(X, Y), expected)
    prepared = spec.prepare(X, block_size=2)
    assert_allclose(prepared["max_diagonal"], np.max(np.diag(spec.evaluate(X, X))))
    assert prepared["evaluation_count"] == len(X)
    assert prepared["validation"] == "analytic_psd_family"


def test_scale_auto_constant_and_frozen_width():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    scale = resolve_kernel(X, gamma="scale")
    assert scale.gamma == pytest.approx(1 / (X.shape[1] * X.var()))
    assert scale.kws["gamma"] == scale.gamma
    assert resolve_kernel(X, gamma="auto").gamma == 0.5
    assert resolve_kernel(np.ones((3, 2)), gamma="scale").gamma == 1.0
    saved = scale.gamma
    scale.evaluate(100 * X, -100 * X)
    assert scale.gamma == saved
    with pytest.raises(ValueError, match="underflow"):
        resolve_kernel([[0.0], [1e-200]], gamma="scale")
    with pytest.raises(ValueError, match="overflow"):
        resolve_kernel([[1e308], [-1e308]], gamma="scale")


@pytest.mark.parametrize("name", ["rbf", "laplacian", "chi2", "poly", "polynomial", "sigmoid"])
def test_zero_gamma_constant_kernel(name):
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    spec = resolve_kernel(X, name, gamma=0)
    constant = np.tanh(1.0) if name == "sigmoid" else 1.0
    assert_allclose(spec.evaluate(X, X), np.full((2, 2), constant))
    assert_allclose(spec.prepare(X)["max_diagonal"], constant)


def test_polynomial_degree_zero_and_stable_cosine_large_offset():
    X = np.array([[1e308, 1e308], [0.0, 0.0]])
    polynomial = resolve_kernel(X, "poly", gamma=1e308, kernel_kws={"degree": 0, "coef0": 0})
    assert_array_equal(polynomial.evaluate(X, X), np.ones((2, 2)))
    assert polynomial.prepare(X)["max_diagonal"] == 1
    cosine = resolve_kernel(X, "cosine")
    assert_allclose(cosine.evaluate(X, X), [[1.0, 0.0], [0.0, 0.0]])
    assert cosine.prepare(np.zeros((2, 2)))["max_diagonal"] == 0


def test_callable_contract_keywords_are_copied_and_top_level_gamma_not_forwarded():
    X = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 3.0]])
    Y = X[:2]
    offset = np.array([0.5])
    kws = {"offset": offset, "gamma": 0.25}
    spec = resolve_kernel(X, linear_with_offset, gamma=-123, kernel_kws=kws)
    assert spec.gamma is None
    assert_allclose(spec.evaluate(X, Y), 0.25 * (X @ Y.T) + 0.5)
    kws["gamma"] = 20
    offset[0] = 99
    assert_allclose(spec.evaluate(X, Y), 0.25 * (X @ Y.T) + 0.5)
    prepared = spec.prepare(X, block_size=2)
    assert prepared["evaluation_count"] == len(X) ** 2
    assert prepared["validation"] == "training_gram_psd_check"
    assert_allclose(prepared["max_diagonal"], np.max(0.25 * np.sum(X * X, axis=1) + 0.5))
    assert not any(isinstance(value, np.ndarray) for value in vars(spec).values())
    restored = pickle.loads(pickle.dumps(spec))
    assert_allclose(restored.evaluate(X, Y), spec.evaluate(X, Y))


def test_mutating_callable_cannot_modify_inputs_or_resolved_keywords():
    def mutating(X, Y, offset):
        X[:] = 1.0
        Y[:] = 2.0
        offset[:] = 3.0
        return X @ Y.T + offset[0]

    X, Y = np.array([[4.0, 5.0]]), np.array([[6.0, 7.0]])
    original_X, original_Y = X.copy(), Y.copy()
    spec = resolve_kernel(X, mutating, kernel_kws={"offset": np.array([0.5])})
    spec.evaluate(X, Y)
    assert_array_equal(X, original_X)
    assert_array_equal(Y, original_Y)
    assert_array_equal(spec.kws["offset"], [0.5])


def test_sigmoid_and_additive_chi2_are_checked_instead_of_assumed_psd():
    # One negative diagonal is already incompatible with any RKHS feature map.
    sigmoid = resolve_kernel([[0.0], [1.0]], "sigmoid", gamma=0, kernel_kws={"coef0": -1})
    with pytest.raises(ValueError, match="nonnegative diagonal"):
        sigmoid.prepare([[0.0], [1.0]])
    # sklearn's additive chi2 is the NEGATIVE distance, not the positive additive kernel.
    additive = resolve_kernel([[0.0], [1.0]], "additive_chi2")
    with pytest.raises(ValueError, match="positive semidefinite"):
        additive.prepare([[0.0], [1.0]])


def test_callable_symmetry_psd_and_shape_rejections():
    X = np.array([[0.0], [1.0]])
    asymmetric = resolve_kernel(X, lambda A, B: A[:, :1] - B[:, :1].T)
    with pytest.raises(ValueError, match="symmetric"):
        asymmetric.prepare(X, block_size=1)
    indefinite = resolve_kernel(X, lambda A, B: np.where(A[:, :1] == B[:, :1].T, 1.0, 2.0))
    with pytest.raises(ValueError, match="positive semidefinite"):
        indefinite.prepare(X)
    wrong_shape = resolve_kernel(X, lambda A, B: np.ones((len(B), len(A))))
    with pytest.raises(ValueError, match="shape"):
        wrong_shape.evaluate(X, X[:1])
    nonfinite = resolve_kernel(X, lambda A, B: np.full((len(A), len(B)), np.nan))
    with pytest.raises(FloatingPointError, match="finite"):
        nonfinite.evaluate(X, X)


def test_near_psd_roundoff_does_not_modify_kernel_values():
    X = np.array([[0.0], [1.0]])
    spec = resolve_kernel(X, lambda A, B: np.where(A[:, :1] == B[:, :1].T, 1.0, 1.0 + 1e-15))
    before = spec.evaluate(X, X)
    spec.prepare(X)
    assert_array_equal(spec.evaluate(X, X), before)


@pytest.mark.parametrize("scale", [1.0, 1e-20, 1e-300, 1e-320, 1e200])
@pytest.mark.parametrize("matrix,error", [
    ([[1.0, 2.0], [2.0, 1.0]], "positive semidefinite"),
    ([[1.0, 0.25], [0.75, 1.0]], "symmetric"),
])
def test_training_kernel_rejection_is_invariant_to_positive_scale(scale, matrix, error):
    X = np.array([[0.0], [1.0]])
    values = scale * np.asarray(matrix)

    def kernel(A, B):
        return values[np.ix_(A[:, 0].astype(int), B[:, 0].astype(int))]

    with pytest.raises(ValueError, match=error):
        resolve_kernel(X, kernel).prepare(X, block_size=1)


@pytest.mark.parametrize("scale", [0.0, 1.0, 1e-20, 1e-300, 1e-320, 1e200])
def test_valid_tiny_and_zero_psd_kernels_preserve_diagonal_and_evaluations(scale):
    X = np.array([[0.0], [1.0]])
    values = scale * np.array([[1.0, 0.5], [0.5, 1.0]])

    def kernel(A, B):
        return values[np.ix_(A[:, 0].astype(int), B[:, 0].astype(int))]

    spec = resolve_kernel(X, kernel)
    before = spec.evaluate(X, X)
    prepared = spec.prepare(X, block_size=1)
    assert prepared["max_diagonal"] == scale
    assert prepared["evaluation_count"] == len(X) ** 2
    assert_array_equal(spec.evaluate(X, X), before)
    assert_array_equal(values, before)


def test_preparation_deadline_checks_surround_blocks_and_eigensolve(monkeypatch):
    X = np.array([[0.0], [1.0], [2.0]])
    events = []
    original = np.linalg.eigvalsh

    def wrapped(matrix):
        events.append("eigensolve")
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigvalsh", wrapped)
    spec = resolve_kernel(X, linear_with_offset, kernel_kws={"offset": 1})
    spec.prepare(X, block_size=2, check_deadline=lambda: events.append("deadline"))
    position = events.index("eigensolve")
    assert events[position - 1] == "deadline"
    assert events[position + 1] == "deadline"
    assert events.count("deadline") >= 10

    class Deadline(Exception):
        pass

    def expired():
        raise Deadline

    with pytest.raises(Deadline) as error:
        resolve_kernel(X).prepare(X, check_deadline=expired)
    assert error.value.evaluation_count == 0


def test_deadline_exception_reports_successful_partial_kernel_evaluations():
    X = np.array([[0.0], [1.0], [2.0]])
    block_finished = False

    def kernel(A, B):
        nonlocal block_finished
        block_finished = True
        return A @ B.T + 1.0

    def deadline():
        if block_finished:
            raise TimeoutError("deadline after block")

    spec = resolve_kernel(X, kernel)
    with pytest.raises(TimeoutError) as error:
        spec.prepare(X, 2, deadline)
    assert error.value.evaluation_count == 2 * len(X)


@pytest.mark.parametrize("gamma", [-1, True, np.nan, np.inf, "unknown"])
def test_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        resolve_kernel([[0.0], [1.0]], gamma=gamma)


@pytest.mark.parametrize("kernel,kws", [
    ("rbf", {"gamma": 1}), ("rbf", {"sigma": 1}), ("rbf", {"bandwidth": 1}),
    ("poly", {"degree": 1.5}), ("poly", {"degree": -1}),
    ("poly", {"coef0": -1}), ("poly", {"degree": True}),
    ("linear", {"n_jobs": 4}), ("precomputed", None), ("unknown", None),
])
def test_invalid_kernel_domains_and_keywords(kernel, kws):
    with pytest.raises(ValueError):
        resolve_kernel([[0.0], [1.0]], kernel, kernel_kws=kws)


def test_chi2_domain_is_checked_on_fit_and_prediction_and_inputs_are_unchanged():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    saved = X.copy()
    with pytest.raises(ValueError, match="nonnegative"):
        resolve_kernel([[-1.0]], "chi2")
    spec = resolve_kernel(X, "chi2")
    with pytest.raises(ValueError, match="nonnegative"):
        spec.evaluate([[-1.0, 0.0]], X)
    spec.evaluate(X, X)
    spec.prepare(X)
    assert_array_equal(X, saved)
    assert resolve_kernel(X, "linear").gamma is None
    assert spec.evaluate(np.empty((0, 2)), X).shape == (0, 2)
