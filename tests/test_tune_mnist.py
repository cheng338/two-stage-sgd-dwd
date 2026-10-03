"""Runner checkpoint and selection contracts using synthetic scores, no fits."""

from contextlib import ExitStack, contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal


_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_SPEC = importlib.util.spec_from_file_location("tune_mnist_test_subject", _EXAMPLES / "tune_mnist.py")
_RUNNER = importlib.util.module_from_spec(_SPEC)
with patch.object(sys, "path", [str(_EXAMPLES), *sys.path]):
    _SPEC.loader.exec_module(_RUNNER)


def candidate(lambd, gamma=2.0, retention=0.05):
    return {"kernel": "rbf", "lambd": float(lambd), "gamma": float(gamma),
            "retention_fraction": retention}


def tiny_data():
    return SimpleNamespace(
        X=np.arange(16, dtype=float).reshape(8, 2) / 255.0,
        y=np.array([-1] * 4 + [1] * 4),
        ids=np.array([f"train:{i}" for i in range(8)]),
        digits=(7, 8), hashes={"images": "synthetic-image-hash", "labels": "synthetic-label-hash"},
    )


def arguments(directory, **overrides):
    values = dict(
        data_dir=Path("never-opened-synthetic-data"), digits=(7, 8), train_size=8,
        folds=2, epochs=3, random_state=17, threads=1, broad_grid=False,
        max_candidates=10, max_expansions=1, refinement_rounds=1, top_k=2,
        output_dir=directory / "run", run=True, resume=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeScores:
    """Capture evaluator boundaries without constructing an estimator."""

    def __init__(self, scores=None, interrupt_once=None, ineligible=False, ineligible_lambdas=()):
        self.scores = scores or {1.0: 0.8, 10.0: 0.7}
        self.interrupt_once = interrupt_once
        self.ineligible = ineligible
        self.ineligible_lambdas = set(ineligible_lambdas)
        self.constructed = []
        self.calls = []

    def factory(self, X, y, ids, folds, **kwargs):
        self.constructed.append((X.copy(), y.copy(), ids.copy(),
                                 [(fit.copy(), score.copy()) for fit, score in folds], kwargs.copy()))
        return SimpleNamespace(evaluate=self.evaluate)

    def evaluate(self, params):
        self.calls.append(copy.deepcopy(params))
        if params["gamma"] == "scale":
            score, eligible = 1.0, True
        else:
            if params["lambd"] == self.interrupt_once:
                self.interrupt_once = None
                raise RuntimeError("synthetic interruption before candidate commit")
            eligible = not self.ineligible and params["lambd"] not in self.ineligible_lambdas
            score = self.scores.get(params["lambd"], 0.5) if eligible else None
        return {"params": copy.deepcopy(params), "mean_score": score, "eligible": eligible,
                "fold_scores": [score, score]}

    @property
    def search_calls(self):
        return [params for params in self.calls if params["gamma"] != "scale"]


@contextmanager
def synthetic_run(fake, *, data=None, expansions=(), refinements=(), boundaries=None):
    data = tiny_data() if data is None else data
    grid = {"kernel": ["rbf"], "lambd": [1.0, 10.0], "gamma": [2.0],
            "retention_fraction": [0.05]}
    with ExitStack() as stack:
        loader = stack.enter_context(patch.object(_RUNNER, "load_mnist_pair", return_value=data))
        stack.enter_context(patch.object(_RUNNER, "mnist_param_grid", return_value=grid))
        stack.enter_context(patch.object(_RUNNER, "FoldEvaluator", side_effect=fake.factory))
        stack.enter_context(patch.object(_RUNNER, "propose_expansion", return_value=list(expansions)))
        stack.enter_context(patch.object(_RUNNER, "propose_refinement", return_value=list(refinements)))
        stack.enter_context(patch.object(_RUNNER, "boundary_axes", return_value=(
            {"lambd": [], "gamma": []} if boundaries is None else boundaries)))
        stack.enter_context(patch("builtins.print"))
        yield loader


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(directory):
    return [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]


class TuningRunnerTests(unittest.TestCase):
    def test_dry_run_never_loads_data_or_constructs_evaluator(self):
        fake = FakeScores()
        with synthetic_run(fake) as loader:
            _RUNNER.main([])
        loader.assert_not_called()
        self.assertEqual(fake.constructed, [])
        self.assertEqual(fake.calls, [])

    def test_initial_grid_budget_is_checked_before_loading_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary), max_candidates=1)
            fake = FakeScores()
            with synthetic_run(fake) as loader, self.assertRaisesRegex(ValueError, "complete initial grid"):
                _RUNNER.run_search(args)
            loader.assert_not_called()
            self.assertFalse(args.output_dir.exists())
            self.assertEqual(fake.calls, [])

    def test_candidate_budget_never_partially_starts_expansion_or_refinement(self):
        proposals = [candidate(0.1), candidate(100.0)]
        for stage in ("expansion", "refinement"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                args = arguments(Path(temporary), max_candidates=3,
                                 max_expansions=int(stage == "expansion"),
                                 refinement_rounds=int(stage == "refinement"))
                fake = FakeScores()
                settings = {"expansions" if stage == "expansion" else "refinements": proposals}
                with synthetic_run(fake, **settings):
                    result = _RUNNER.run_search(args)
                self.assertEqual(fake.search_calls, [candidate(1), candidate(10)])
                self.assertEqual(result["attempted_candidates"], 2)
                self.assertIn(f"candidate_budget_prevents_complete_{stage}", result["review_reasons"])
                self.assertEqual([row["name"] for row in read_json(args.output_dir / "stages.json")], ["coarse"])
                self.assertEqual(len(read_records(args.output_dir)), 2)

    def test_prior_winner_remains_eligible_and_control_never_competes(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            fake = FakeScores({1.0: 0.8, 10.0: 0.7, 0.1: 0.6, 100.0: 0.55, 3.0: 0.65})
            with synthetic_run(fake, expansions=[candidate(0.1), candidate(100)], refinements=[candidate(3)]):
                result = _RUNNER.run_search(args)
            self.assertEqual(result["params"], candidate(1))
            self.assertEqual(result["mean_cv_accuracy"], 0.8)
            self.assertEqual(result["eligible_candidates"], 5)
            self.assertEqual(result["attempted_candidates"], 5)
            self.assertEqual(len(fake.calls), 6)  # The separate inherited-default control.
            control = read_json(args.output_dir / "inherited-default-control.json")
            self.assertEqual(control["mean_score"], 1.0)
            self.assertEqual(control["params"]["retention_fraction"], 0.3)
            self.assertEqual(control["params"]["retention_fraction"],
                             _RUNNER.two_stage_sgd_dwd.TwoStageDWDClassifier().retention_fraction)
            self.assertTrue(all(row["params"]["gamma"] != "scale" for row in read_records(args.output_dir)))
            self.assertTrue(all(row["params"]["retention_fraction"] == 0.05
                                for row in read_records(args.output_dir)))
            stages = read_json(args.output_dir / "stages.json")
            self.assertEqual([row["score_after"] for row in stages], [0.8, 0.8, 0.8])

    def test_resume_finishes_pending_stage_without_repeating_committed_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            fake = FakeScores(interrupt_once=10.0)
            with synthetic_run(fake):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    _RUNNER.run_search(args)
                committed = read_records(args.output_dir)
                self.assertEqual([row["params"] for row in committed], [candidate(1)])
                self.assertFalse(read_json(args.output_dir / "stages.json")[0]["completed"])
                saved_splits = read_json(args.output_dir / "splits.json")
                args.resume = True
                result = _RUNNER.run_search(args)
                completed_calls = copy.deepcopy(fake.calls)
                completed_constructions = len(fake.constructed)
                self.assertEqual(_RUNNER.run_search(args), result)
            self.assertEqual(fake.calls, completed_calls)
            self.assertEqual(len(fake.constructed), completed_constructions)
            self.assertEqual(sum(params["gamma"] == "scale" for params in fake.calls), 1)
            self.assertEqual(fake.search_calls, [candidate(1), candidate(10), candidate(10)])
            self.assertEqual(read_records(args.output_dir)[0], committed[0])
            self.assertEqual(len(read_records(args.output_dir)), 2)
            self.assertTrue(read_json(args.output_dir / "stages.json")[0]["completed"])
            self.assertEqual(read_json(args.output_dir / "splits.json"), saved_splits)
            self.assertEqual(result["params"], candidate(1))
            for original, resumed in zip(fake.constructed[0][:3], fake.constructed[1][:3]):
                assert_array_equal(original, resumed)
            for first, second in zip(fake.constructed[0][3], fake.constructed[1][3]):
                assert_array_equal(first[0], second[0])
                assert_array_equal(first[1], second[1])

    def test_unexpected_candidate_failure_is_recorded_and_only_retried_on_resume(self):
        for error in (FloatingPointError("synthetic numerical failure"), RuntimeError("synthetic evaluator failure")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temporary:
                args = arguments(Path(temporary))
                fake = FakeScores()
                original_evaluate = fake.evaluate
                failed = False

                def fail_once(params):
                    nonlocal failed
                    if not failed and params["gamma"] != "scale" and params["lambd"] == 10.0:
                        failed = True
                        raise error
                    return original_evaluate(params)

                with synthetic_run(fake), patch.object(fake, "evaluate", side_effect=fail_once):
                    with self.assertRaises(type(error)) as caught:
                        _RUNNER.run_search(args)
                    self.assertIs(caught.exception, error)
                    failure = read_json(args.output_dir / "failure.json")
                    self.assertEqual(failure["stage"], "coarse")
                    self.assertEqual(failure["params"], candidate(10))
                    self.assertEqual(failure["error_type"], type(error).__name__)
                    self.assertEqual(failure["message"], str(error))
                    self.assertEqual(failure["completed_candidates"], 1)
                    self.assertEqual(read_json(args.output_dir / "status.json")["state"], "failed")
                    committed = read_records(args.output_dir)
                    self.assertEqual([row["params"] for row in committed], [candidate(1)])
                    self.assertTrue(committed[0]["eligible"])
                    self.assertFalse(read_json(args.output_dir / "stages.json")[0]["completed"])
                    self.assertEqual(read_json(args.output_dir / "plan_state.json")["phase"], "coarse")
                    self.assertFalse((args.output_dir / "selection.json").exists())
                    args.resume = True
                    result = _RUNNER.run_search(args)
                self.assertEqual(result["attempted_candidates"], 2)
                self.assertEqual(read_records(args.output_dir)[0], committed[0])
                self.assertEqual(read_json(args.output_dir / "status.json")["state"], "exploration_completed")
                self.assertEqual(read_json(args.output_dir / "failure.json"), failure)
                self.assertTrue(all(row["eligible"] for row in read_records(args.output_dir)))

    def test_unexpected_inherited_control_failure_is_recorded_and_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            fake = FakeScores()
            original_evaluate = fake.evaluate
            error = FloatingPointError("synthetic inherited-control failure")
            failed = False

            def fail_control_once(params):
                nonlocal failed
                if not failed and params["gamma"] == "scale":
                    failed = True
                    raise error
                return original_evaluate(params)

            with synthetic_run(fake), patch.object(fake, "evaluate", side_effect=fail_control_once):
                with self.assertRaises(FloatingPointError) as caught:
                    _RUNNER.run_search(args)
                self.assertIs(caught.exception, error)
                failure = read_json(args.output_dir / "failure.json")
                self.assertEqual(failure["stage"], "inherited-default-control")
                self.assertEqual(failure["params"], {
                    "kernel": "rbf", "lambd": 1e-6, "gamma": "scale", "retention_fraction": 0.3,
                })
                self.assertEqual(failure["error_type"], "FloatingPointError")
                self.assertEqual(failure["message"], str(error))
                self.assertEqual(failure["completed_candidates"], 0)
                self.assertEqual(read_json(args.output_dir / "status.json")["state"], "failed")
                self.assertEqual(read_records(args.output_dir), [])
                self.assertEqual(read_json(args.output_dir / "stages.json"), [])
                self.assertEqual(read_json(args.output_dir / "plan_state.json")["phase"], "coarse")
                self.assertFalse((args.output_dir / "inherited-default-control.json").exists())
                self.assertFalse((args.output_dir / "selection.json").exists())
                args.resume = True
                result = _RUNNER.run_search(args)
            self.assertEqual(result["attempted_candidates"], 2)
            self.assertTrue(read_json(args.output_dir / "inherited-default-control.json")["eligible"])
            self.assertEqual(read_json(args.output_dir / "status.json")["state"], "exploration_completed")
            self.assertEqual(read_json(args.output_dir / "failure.json"), failure)

    def test_resume_during_refinement_preserves_prior_expansion_stop_decision(self):
        # An interior coarse winner stops expansion. Refinement then exposes
        # a new best boundary; resuming must not revisit an earlier phase that
        # the uninterrupted execution has already left.
        def expansion_after_refinement(records, retention_values):
            if any(row.get("stage", "").startswith("refinement") for row in records):
                return [candidate(1000)]
            return []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores = {1.0: 0.8, 10.0: 0.7, 3.0: 0.9, 4.0: 0.85, 1000.0: 0.95}
            reference_args = arguments(root, output_dir=root / "uninterrupted")
            reference_fake = FakeScores(scores)
            with synthetic_run(reference_fake, refinements=[candidate(3), candidate(4)]):
                with patch.object(_RUNNER, "propose_expansion", side_effect=expansion_after_refinement):
                    reference = _RUNNER.run_search(reference_args)
            interrupted_args = arguments(root, output_dir=root / "interrupted")
            interrupted_fake = FakeScores(scores, interrupt_once=4.0)
            with synthetic_run(interrupted_fake, refinements=[candidate(3), candidate(4)]):
                with patch.object(_RUNNER, "propose_expansion", side_effect=expansion_after_refinement):
                    with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                        _RUNNER.run_search(interrupted_args)
                    interrupted_args.resume = True
                    resumed = _RUNNER.run_search(interrupted_args)
            self.assertEqual(reference["params"], candidate(3))
            self.assertEqual(resumed, reference)
            self.assertEqual([row["params"] for row in read_records(interrupted_args.output_dir)],
                             [row["params"] for row in read_records(reference_args.output_dir)])
            self.assertFalse(any(params["lambd"] == 1000.0 for params in interrupted_fake.search_calls))

    def test_resume_after_completed_stage_before_cursor_write_counts_plateau_once(self):
        def proposals(records, retention_values):
            attempted = {row["params"]["lambd"] for row in records}
            for value in (0.1, 0.01, 0.001):
                if value not in attempted:
                    return [candidate(value)]
            return []

        real_write = _RUNNER.write_json
        interrupted = False

        def interrupted_write(path, value):
            nonlocal interrupted
            if (not interrupted and path.name == "plan_state.json"
                    and value["phase"] == "expansion" and value["index"] == 1):
                interrupted = True
                raise RuntimeError("synthetic interruption before cursor commit")
            return real_write(path, value)

        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary), max_expansions=4, refinement_rounds=0)
            fake = FakeScores()
            with synthetic_run(fake, boundaries={"lambd": ["lower"], "gamma": []}):
                with patch.object(_RUNNER, "propose_expansion", side_effect=proposals):
                    with patch.object(_RUNNER, "write_json", side_effect=interrupted_write):
                        with self.assertRaisesRegex(RuntimeError, "before cursor commit"):
                            _RUNNER.run_search(args)
                    self.assertTrue(read_json(args.output_dir / "stages.json")[-1]["completed"])
                    self.assertEqual(read_json(args.output_dir / "plan_state.json")["index"], 0)
                    args.resume = True
                    result = _RUNNER.run_search(args)
            self.assertEqual(fake.search_calls, [candidate(1), candidate(10), candidate(0.1), candidate(0.01)])
            self.assertEqual(result["attempted_candidates"], 4)
            self.assertIn("boundary_plateau_at_this_epoch_budget", result["review_reasons"])
            plan = read_json(args.output_dir / "plan_state.json")
            self.assertEqual(plan["phase"], "completed")
            self.assertEqual(plan["nonimproving"], 2)
            self.assertTrue(plan["plateau"])

    def test_resume_rejects_changed_training_budget_or_source_hashes(self):
        for change in ("epochs", "data_hashes"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                args = arguments(Path(temporary))
                data, fake = tiny_data(), FakeScores(interrupt_once=10.0)
                with synthetic_run(fake, data=data):
                    with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                        _RUNNER.run_search(args)
                    before = (args.output_dir / "records.jsonl").read_bytes()
                    calls = copy.deepcopy(fake.calls)
                    args.resume = True
                    if change == "epochs":
                        args.epochs += 1
                    else:
                        data.hashes["images"] = "different-source-content"
                    with self.assertRaisesRegex(ValueError, "[Rr]esume|configuration|differs"):
                        _RUNNER.run_search(args)
                self.assertEqual(fake.calls, calls)
                self.assertEqual((args.output_dir / "records.jsonl").read_bytes(), before)

    def test_resume_rejects_changed_grid_even_with_the_same_candidate_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            fake = FakeScores(interrupt_once=10.0)
            with synthetic_run(fake):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    _RUNNER.run_search(args)
                config = read_json(args.output_dir / "config.json")
                self.assertEqual(config["grid"]["lambd"], [1.0, 10.0])
                self.assertTrue(config["package_version"])
                self.assertTrue(config["implementation_sha256"])
                replacement = copy.deepcopy(config["grid"])
                replacement["lambd"] = [1.0, 11.0]
                calls = copy.deepcopy(fake.calls)
                args.resume = True
                with patch.object(_RUNNER, "mnist_param_grid", return_value=replacement):
                    with self.assertRaisesRegex(ValueError, "[Rr]esume|configuration|differs"):
                        _RUNNER.run_search(args)
            self.assertEqual(fake.calls, calls)

    def test_new_run_snapshots_recorded_implementation_and_resume_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            fake = FakeScores()
            with synthetic_run(fake):
                _RUNNER.run_search(args)
                config = read_json(args.output_dir / "config.json")
                snapshot = args.output_dir / "implementation-snapshot"
                self.assertTrue(config["implementation_sha256"])
                for name, digest in config["implementation_sha256"].items():
                    self.assertEqual(hashlib.sha256((snapshot / name).read_bytes()).hexdigest(), digest)
                (snapshot / "tune_mnist.py").write_bytes(b"unexpected replacement")
                calls = copy.deepcopy(fake.calls)
                args.resume = True
                with self.assertRaisesRegex(ValueError, "[Ii]mplementation snapshot"):
                    _RUNNER.run_search(args)
            self.assertEqual(fake.calls, calls)

    def test_resume_rejects_saved_observation_id_or_fold_mismatch(self):
        for change in ("observation_ids", "folds"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                args = arguments(Path(temporary))
                fake = FakeScores(interrupt_once=10.0)
                with synthetic_run(fake):
                    with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                        _RUNNER.run_search(args)
                    path = args.output_dir / "splits.json"
                    payload = read_json(path)
                    if change == "observation_ids":
                        payload["observation_ids"][0] = "train:unexpected-row"
                    else:
                        payload["folds"] = list(reversed(payload["folds"]))
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    calls = copy.deepcopy(fake.calls)
                    args.resume = True
                    with self.assertRaisesRegex(ValueError, "[Rr]esume|fold|split"):
                        _RUNNER.run_search(args)
                self.assertEqual(fake.calls, calls)

    def test_resume_recovers_only_unterminated_truncated_final_record_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            fake = FakeScores(interrupt_once=10.0)
            with synthetic_run(fake):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    _RUNNER.run_search(args)
                path = args.output_dir / "records.jsonl"
                committed = read_records(args.output_dir)
                damaged = path.read_bytes() + b'{"params":'
                path.write_bytes(damaged)
                args.resume = True
                result = _RUNNER.run_search(args)
            self.assertEqual((args.output_dir / "records-interrupted.jsonl").read_bytes(), damaged)
            records = read_records(args.output_dir)
            self.assertEqual(records[0], committed[0])
            self.assertEqual([row["params"] for row in records], [candidate(1), candidate(10)])
            self.assertEqual(result["attempted_candidates"], 2)

    def test_resume_handles_valid_final_record_missing_only_newline(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            fake = FakeScores(interrupt_once=10.0)
            with synthetic_run(fake):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    _RUNNER.run_search(args)
                path = args.output_dir / "records.jsonl"
                committed = read_records(args.output_dir)[0]
                path.write_bytes(path.read_bytes().rstrip(b"\r\n"))
                args.resume = True
                result = _RUNNER.run_search(args)
            records = read_records(args.output_dir)
            self.assertEqual(records[0], committed)
            self.assertEqual([row["params"] for row in records], [candidate(1), candidate(10)])
            self.assertEqual(result["attempted_candidates"], 2)

    def test_resume_does_not_silently_discard_malformed_complete_or_earlier_records(self):
        for suffix in (b'{"params":\n', b'{"params":\n{}\n'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temporary:
                args = arguments(Path(temporary))
                fake = FakeScores(interrupt_once=10.0)
                with synthetic_run(fake):
                    with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                        _RUNNER.run_search(args)
                    path = args.output_dir / "records.jsonl"
                    damaged = path.read_bytes() + suffix
                    path.write_bytes(damaged)
                    calls = copy.deepcopy(fake.calls)
                    args.resume = True
                    with self.assertRaises((ValueError, json.JSONDecodeError)):
                        _RUNNER.run_search(args)
                self.assertEqual(fake.calls, calls)
                self.assertEqual(path.read_bytes(), damaged)
                self.assertFalse((args.output_dir / "records-interrupted.jsonl").exists())

    def test_refinement_improvement_invalidates_previous_boundary_plateau(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary), max_expansions=2)
            fake = FakeScores({1.0: 0.8, 10.0: 0.7, 0.1: 0.6, 0.01: 0.65, 3.0: 0.9})
            with synthetic_run(fake, refinements=[candidate(3)],
                               boundaries={"lambd": ["lower"], "gamma": []}):
                with patch.object(_RUNNER, "propose_expansion", side_effect=[
                    [candidate(0.1)], [candidate(0.01)],
                ]):
                    result = _RUNNER.run_search(args)
            self.assertEqual(result["params"], candidate(3))
            self.assertIn("unresolved_boundary", result["review_reasons"])
            self.assertNotIn("boundary_plateau_at_this_epoch_budget", result["review_reasons"])

    def test_only_training_data_reaches_evaluator_and_score_folds_are_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            data, fake = tiny_data(), FakeScores()
            original_X, original_y, original_ids = data.X.copy(), data.y.copy(), data.ids.copy()
            with synthetic_run(fake, data=data) as loader:
                result = _RUNNER.run_search(args)
            loader.assert_called_once_with(args.data_dir, args.digits, partition="train")
            X, y, ids, folds, settings = fake.constructed[0]
            assert_array_equal(X, original_X)
            assert_array_equal(y, original_y)
            assert_array_equal(ids, original_ids)
            self.assertTrue(all(value.startswith("train:") for value in ids))
            seen_scores = []
            for fit, score in folds:
                self.assertFalse(set(fit).intersection(score))
                self.assertEqual(set(fit).union(score), set(range(len(y))))
                self.assertEqual(set(y[fit]), {-1, 1})
                self.assertEqual(set(y[score]), {-1, 1})
                seen_scores.extend(score.tolist())
            self.assertEqual(sorted(seen_scores), list(range(len(y))))
            self.assertNotIn("validation_data", settings)
            self.assertFalse(result["official_test_partition_loaded"])
            self.assertFalse(read_json(args.output_dir / "config.json")["official_test_partition_loaded"])
            assert_array_equal(data.X, original_X)
            assert_array_equal(data.y, original_y)
            assert_array_equal(data.ids, original_ids)

    def test_successful_control_cannot_rescue_all_ineligible_grid_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            fake = FakeScores(ineligible=True)
            with synthetic_run(fake), self.assertRaisesRegex(RuntimeError, "No candidate"):
                _RUNNER.run_search(args)
            self.assertTrue(read_json(args.output_dir / "inherited-default-control.json")["eligible"])
            self.assertFalse(any(row["eligible"] for row in read_records(args.output_dir)))
            self.assertFalse((args.output_dir / "selection.json").exists())

    def test_json_null_scores_for_ineligible_candidates_do_not_abort_adaptation(self):
        # Use actual proposal helpers here: the evaluator's JSON-safe failed
        # score representation must survive the boundary/refinement boundary.
        actual_expansion = _RUNNER.propose_expansion
        actual_refinement = _RUNNER.propose_refinement
        actual_boundaries = _RUNNER.boundary_axes
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary), max_candidates=100)
            fake = FakeScores(ineligible_lambdas={10.0})
            with synthetic_run(fake):
                with patch.object(_RUNNER, "propose_expansion", actual_expansion), \
                        patch.object(_RUNNER, "propose_refinement", actual_refinement), \
                        patch.object(_RUNNER, "boundary_axes", actual_boundaries):
                    result = _RUNNER.run_search(args)
            self.assertEqual(result["params"]["lambd"], 1.0)
            self.assertEqual(result["mean_cv_accuracy"], 0.8)
            self.assertGreater(result["attempted_candidates"], 2)
            failed = [row for row in read_records(args.output_dir) if not row["eligible"]]
            self.assertTrue(failed)
            self.assertTrue(all(row["mean_score"] is None for row in failed))


if __name__ == "__main__":
    unittest.main()
