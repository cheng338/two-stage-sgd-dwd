"""Validation-only RBF recovery: arithmetic, checkpoint and work contracts."""
from decimal import Decimal, localcontext
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from two_stage_sgd_dwd import KernelSGDDWD, LinearSGDDWD, TwoStageDWDClassifier
from two_stage_sgd_dwd import _core, _estimators as est, _kernels, _prediction


def _arguments(gamma=1e-30):
    return ("kernel", np.array([[-1.], [-1e-6], [0.], [1e-6], [1.]]),
            np.array([-1., 1.]), 0., np.array([[-1.], [1.]]), gamma)


def _decimal_score(query, basis, coefficients, intercept, gamma):
    # Independent evaluation at exact binary64 inputs, including subnormals.
    with localcontext() as context:
        context.prec = 420
        total = Decimal.from_float(float(intercept))
        width = Decimal.from_float(float(gamma))
        for row, coefficient in zip(basis, coefficients):
            distance = sum(((Decimal.from_float(float(x)) - Decimal.from_float(float(y))) ** 2
                            for x, y in zip(query, row)), Decimal(0))
            total += Decimal.from_float(float(coefficient)) * (-width * distance).exp()
        return +total


@pytest.mark.parametrize("gamma", [1e-12, 1e-30])
@pytest.mark.parametrize("batch", [1, 3, 256])
@pytest.mark.parametrize("readonly", [False, True])
def test_recovery_matches_analytic_signs_and_preserves_inputs(gamma, batch, readonly):
    args = _arguments(gamma)
    for value in (args[1], args[2], args[4]):
        value.flags.writeable = not readonly
    before = pickle.dumps(args)
    work, logical = est._validation_recovery_counts(), [0]
    actual = est._validation_scores(*args, batch_size=batch, kernel_counter=logical,
                                    recovery_work=work)
    references = [_decimal_score(row, args[4], args[2], args[3], gamma) for row in args[1]]
    np.testing.assert_array_equal(np.sign(actual), [-1, -1, 0, 1, 1])
    reference_values = np.array([float(x) for x in references])
    # The outer 1e-12-width scores exceed the recovery envelope and keep their
    # original exp subtraction error. Only the near-zero scores are repaired.
    np.testing.assert_allclose(actual, reference_values, rtol=5e-5, atol=0)
    np.testing.assert_allclose(actual[1:4], reference_values[1:4], rtol=5e-10, atol=0)
    assert pickle.dumps(args) == before
    assert logical == [len(args[1]) * len(args[4])]
    assert work["eligible_calls"] == 1 and work["delegated_calls"] == 0
    assert work["near_constant_score_rows_completed"] >= 3
    assert work["near_constant_probe_entries_requested"] == work["near_constant_probe_entries_completed"]
    assert work["near_constant_score_entries_completed"] == 2 * work["near_constant_score_rows_completed"]


def test_recovery_is_not_a_universal_error_improvement_at_subnormal_widths():
    # Rounding two scaled distances to one and zero subnormal units can amplify
    # their difference. Keep this adverse case visible rather than claiming
    # the expm1 identity improves every floating-point score.
    gamma = np.finfo(float).smallest_subnormal
    query, basis = np.array([[.05]]), np.array([[-.75], [.75]])
    coefficients = np.array([-1e16, 1e16])
    args = "kernel", query, coefficients, 0., basis, gamma
    work = est._validation_recovery_counts()
    original = est._scores(*args)[0]
    corrected = est._validation_scores(*args, recovery_work=work)[0]
    reference = _decimal_score(query[0], basis, coefficients, 0., gamma)
    with localcontext() as context:
        context.prec = 420
        old_error = abs(Decimal.from_float(float(original)) - reference)
        new_error = abs(Decimal.from_float(float(corrected)) - reference)
    assert original == 0 and corrected > 0
    assert old_error < new_error < Decimal("1e-307")
    assert work["near_constant_score_rows_completed"] == 1


@pytest.mark.parametrize("batch", [1, 4, 256])
def test_ordinary_scores_are_byte_exact_and_resolved_specs_are_eligible(batch):
    rng = np.random.default_rng(627)
    query, basis, coefficients = rng.normal(size=(9, 3)), rng.normal(size=(13, 3)), rng.normal(size=13)
    spec = _kernels.resolve_kernel(basis, "rbf", gamma=.4)
    args = "kernel", query, coefficients, .3, basis, .4, batch
    work, first, second = est._validation_recovery_counts(), [0], [0]
    old = est._scores(*args, kernel_counter=first, kernel_spec=spec)
    new = est._validation_scores(*args, kernel_counter=second, kernel_spec=spec, recovery_work=work)
    assert old.tobytes() == new.tobytes() and first == second == [9 * 13]
    assert work["eligible_calls"] == 1 and work["near_constant_probe_rows_requested"] == 0


@pytest.mark.parametrize("target", ["scores", "kernel_values", "estimator_rbf", "core_rbf",
                                    "kernel_rbf", "core_distance", "prediction_distance", "repair"])
def test_replaced_helpers_preserve_existing_hook_behavior(target, monkeypatch):
    args = _arguments()
    locations = {"scores": (est, "_scores"), "kernel_values": (est, "_kernel_values"),
                 "estimator_rbf": (est, "rbf_kernel"), "core_rbf": (_core, "rbf_kernel"),
                 "kernel_rbf": (_kernels, "rbf_kernel"), "core_distance": (_core, "cdist"),
                 "prediction_distance": (_prediction, "cdist"),
                 "repair": (_prediction, "_near_constant_scores")}
    module, attribute = locations[target]
    original = getattr(module, attribute)
    calls = []

    def replacement(*a, **kw):
        calls.append(target)
        if target == "repair":
            raise AssertionError("Foreign repair must not be used in validation")
        result = original(*a, **kw)
        return result + .125 if target == "scores" else result

    monkeypatch.setattr(module, attribute, replacement)
    work = est._validation_recovery_counts()
    expected = est._scores(*args)
    actual = est._validation_scores(*args, recovery_work=work)
    assert actual.tobytes() == expected.tobytes()
    assert work["delegated_calls"] == 1 and work["eligible_calls"] == 0
    if target == "scores":
        assert calls == [target, target]
    if target == "repair":
        assert not calls


@pytest.mark.parametrize("case", ["strided", "fortran", "subclass", "float32", "linear",
                                  "callable", "cosine", "mismatched_gamma", "extra_keyword",
                                  "numpy_counter"])
def test_unqualified_inputs_delegate_without_new_behavior(case):
    args = list(_arguments())
    # Two features make the Fortran/strided distinction meaningful.
    args[1], args[4] = np.repeat(args[1], 2, axis=1), np.repeat(args[4], 2, axis=1)
    spec, count = None, [0]
    if case == "strided":
        args[1] = np.repeat(args[1], 2, axis=1)[:, ::2]
    elif case == "fortran":
        args[1] = np.asfortranarray(args[1])
    elif case == "subclass":
        class CustomArray(np.ndarray):
            pass
        args[1] = args[1].view(CustomArray)
    elif case == "float32":
        args[1] = args[1].astype(np.float32)
    elif case == "linear":
        args[0] = "linear"
    elif case == "callable":
        spec = _kernels.KernelSpec(lambda x, y: x @ y.T, None, {})
    elif case == "cosine":
        spec = _kernels.resolve_kernel(args[4], "cosine")
    elif case == "mismatched_gamma":
        spec = _kernels.KernelSpec("rbf", args[5] * 2, {})
    elif case == "extra_keyword":
        spec = _kernels.KernelSpec("rbf", args[5], {"unexpected": 1})
    else:
        count = np.array([0], dtype=np.int64)
    work = est._validation_recovery_counts()
    first, second = count.copy(), count.copy()
    expected = est._scores(*args, kernel_spec=spec, kernel_counter=first)
    actual = est._validation_scores(*args, kernel_spec=spec, kernel_counter=second, recovery_work=work)
    assert expected.tobytes() == actual.tobytes() and first[0] == second[0]
    assert work["eligible_calls"] == 0 and work["delegated_calls"] == 1


@pytest.mark.parametrize("boundary", range(1, 8))
def test_deadline_boundaries_and_logical_prefix_are_unchanged(boundary, monkeypatch):
    outcomes = []
    for scorer in (est._scores, est._validation_scores):
        calls, counter = [], [0]

        def check(deadline):
            calls.append(deadline)
            if len(calls) == boundary:
                raise est._Deadline

        monkeypatch.setattr(est, "_check_deadline", check)
        try:
            scorer(*_arguments(), batch_size=2, deadline=10., kernel_counter=counter)
        except est._Deadline:
            status = "deadline"
        else:
            status = "complete"
        outcomes.append((status, len(calls), counter[0]))
    assert outcomes[0] == outcomes[1]


def test_failed_probe_keeps_requested_and_completed_work_separate(monkeypatch):
    args = _arguments()
    work, logical = est._validation_recovery_counts(), [0]
    original = _prediction._finite_result

    def fail_after_distance(value, name):
        if name == "squared Euclidean distances":
            raise FloatingPointError("Injected failure after distance evaluation")
        return original(value, name)

    monkeypatch.setattr(_prediction, "_finite_result", fail_after_distance)
    with pytest.raises(FloatingPointError, match="after distance evaluation"):
        est._validation_scores(*args, batch_size=2, kernel_counter=logical, recovery_work=work)
    assert logical == [4]
    assert work["near_constant_probe_entries_requested"] == 2
    assert work["near_constant_probe_entries_completed"] == 0
    assert work["near_constant_score_entries_requested"] == 0


@pytest.mark.parametrize("policy", ["ignore", "warn", "raise"])
def test_invalid_second_block_preserves_failure_prefix_and_caller_policy(policy):
    args = list(_arguments())
    args[1][3, 0] = np.nan
    for scorer in (est._scores, est._validation_scores):
        counter = [0]
        with np.errstate(all=policy):
            before = np.geterr()
            with pytest.raises(ValueError, match="finite"):
                scorer(*args, batch_size=2, kernel_counter=counter)
            assert np.geterr() == before
        assert counter == [4]


def _legacy_validation(*args, recovery_work=None, **kwargs):
    return est._scores(*args, **kwargs)


def test_material_checkpoint_improvement_uses_unchanged_selection_rules(monkeypatch):
    basis, query = np.array([[-1.], [1.]]), np.array([[.25]])
    states = [np.array([-1e12, 1e12]), np.array([-2e12, 2e12])]
    settings = SimpleNamespace(batch_size=1, sampling="with_replacement", lambd_=1e-6,
        max_kernel_diagonal_=1., max_epochs=3, gamma_=1e-18, _kernel_spec_=None,
        prediction_batch_size=2, validation_interval=1, tol=0., patience=1,
        min_epochs=1, stopping_="validation")
    outcomes = []
    for legacy in (True, False):
        steps = [0]

        def prescribed(*args):
            value = states[min(steps[0] // 2, 1)].copy()
            steps[0] += 1
            return value, 0.

        with monkeypatch.context() as context:
            context.setattr(est, "kernel_step", prescribed)
            if legacy:
                context.setattr(est, "_validation_scores", _legacy_validation)
            result = est._optimize_steps("kernel", basis, np.array([-1., 1.]),
                (query, np.ones(1)), np.random.default_rng(211), settings, None)
        outcomes.append(result)
    old, new = outcomes
    assert old["selected_iteration_"] == 2 and old["n_updates_"] == 4
    assert new["selected_iteration_"] == 4 and new["n_updates_"] == 6
    assert old["best_validation_loss_"] == 1.
    assert 5e-7 < new["history_"][1]["validation_loss"] - new["best_validation_loss_"] < 2e-6
    assert new["history_"][2]["patience_misses"] == 0
    assert new["history_"][3]["patience_misses"] == 1
    assert old["stop_reason_"] == new["stop_reason_"] == "validation_patience"
    # These are prescribed checkpoint states, not naturally reached SGD fits.


@pytest.mark.parametrize("batch", [1, 2, "full"])
def test_validation_recovery_does_not_change_update_or_self_norm_arithmetic(batch, monkeypatch):
    X, y = np.array([[-3.], [-1.], [1.], [3.]]), np.array([-1, -1, 1, 1])
    Q, labels = np.array([[-2.], [-.25], [.25], [2.]]), np.array([-1, -1, 1, 1])
    params = dict(gamma=1e-30, lambd=1e-12, max_epochs=8, batch_size=batch,
                  stopping="fixed", validation_interval=1, random_state=3,
                  optimizer_backend="python", prediction_backend="direct")
    with monkeypatch.context() as context:
        context.setattr(est, "_validation_scores", _legacy_validation)
        old = KernelSGDDWD(**params).fit(X, y, validation_data=(Q, labels))
    new = KernelSGDDWD(**params).fit(X, y, validation_data=(Q, labels))
    assert pickle.dumps(old.final_state_) == pickle.dumps(new.final_state_)
    assert pickle.dumps(old.rng_state_) == pickle.dumps(new.rng_state_)
    assert old.kernel_evaluation_count_ == new.kernel_evaluation_count_
    assert new.validation_recovery_work_["eligible_calls"] == 8
    # Self scores must keep the original formula even when validation recovers
    # a tiny residual. Do not assert identical checkpoint choice near ties.
    expected = _core.rbf_kernel(new.basis_, new.basis_, new.gamma_) @ new.dual_coef_[0]
    actual = est._scores("kernel", new.basis_, new.dual_coef_[0], 0., new.basis_, new.gamma_)
    np.testing.assert_array_equal(actual, expected)


def test_work_reports_are_local_serializable_and_exposed_for_both_stages():
    X, y = np.array([[-3.], [-1.], [1.], [3.]]), np.array([-1, -1, 1, 1])
    Q, labels = np.array([[-.25], [.25]]), np.array([-1, 1])
    model = TwoStageDWDClassifier(retention_fraction=1., gamma=1e-30,
        max_epochs=3, validation_interval=1, stopping="fixed", batch_size="full",
        optimizer_backend="python", random_state=7).fit(X, y, validation_data=(Q, labels))
    linear = model.linear_estimator_.validation_recovery_work_
    kernel = model.kernel_estimator_.validation_recovery_work_
    assert linear["delegated_calls"] == 3 and kernel["eligible_calls"] == 3
    assert model.stage_diagnostics_["linear"]["validation_recovery_work"] == linear
    assert model.stage_diagnostics_["kernel"]["validation_recovery_work"] == kernel
    assert model.stage_diagnostics_["kernel"]["validation_recovery_work"] is not kernel
    restored = pickle.loads(pickle.dumps(model))
    assert restored.kernel_estimator_.validation_recovery_work_ == kernel
    empty = LinearSGDDWD(max_epochs=1, optimizer_backend="python").fit(X, y).validation_recovery_work_
    assert all(value == 0 for value in empty.values())


def test_completed_recovery_work_remains_visible_after_callback_failure():
    X, y = np.array([[-3.], [-1.], [1.], [3.]]), np.array([-1, -1, 1, 1])
    Q, labels = np.array([[-.25], [.25]]), np.array([-1, 1])
    model = KernelSGDDWD(gamma=1e-30, max_epochs=3, validation_interval=1,
                        batch_size="full", optimizer_backend="python", random_state=7)

    def abort(state, entry):
        if state["iteration"]:
            raise RuntimeError("User callback stopped after validation")

    with pytest.raises(RuntimeError, match="User callback"):
        model.fit(X, y, validation_data=(Q, labels), checkpoint_callback=abort)
    assert model.validation_recovery_work_["eligible_calls"] == 1
    assert model.validation_recovery_work_["near_constant_score_rows_completed"] == len(Q)
