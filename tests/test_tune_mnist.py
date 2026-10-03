"""Adaptive runner qualification and durable recovery, with deterministic CV."""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
_SPEC = importlib.util.spec_from_file_location('tune_mnist_test_subject', _EXAMPLES / 'tune_mnist.py')
_RUNNER = importlib.util.module_from_spec(_SPEC)
with patch.object(sys, 'path', [str(_EXAMPLES), *sys.path]):
    _SPEC.loader.exec_module(_RUNNER)


def grid():
    return {'kernel': ['rbf'], 'lambd': [.25, 1., 4.], 'gamma': [.25, 1., 4.],
            'retention_fraction': [.5]}


def arguments(path, **changes):
    values = dict(data_dir=Path('never-opened'), digits=(2, 3), train_size=12,
                  folds=2, epochs=3, random_state=17, threads=1, broad_grid=False,
                  max_candidates=200, max_expansions=3, refinement_rounds=None,
                  stable_refinement_rounds=2, bracket_factor=2., top_k=1,
                  output_dir=path / 'run', run=True, resume=False)
    return SimpleNamespace(**(values | changes))


def read(path):
    return json.loads(path.read_text())


def records(out):
    return [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]


class Scores:
    def __init__(self, mode='peak', fail_once=None, fail_error=None):
        self.mode = mode
        self.fail_once = fail_once
        self.fail_error = fail_error or RuntimeError('synthetic interruption')
        self.calls, self.constructed = [], []

    def factory(self, X, y, ids, folds, **controls):
        self.constructed.append((X.copy(), y.copy(), ids.copy(), copy.deepcopy(folds), controls))
        return SimpleNamespace(evaluate=self.evaluate)

    def evaluate(self, params):
        self.calls.append(copy.deepcopy(params))
        key = 'control' if params['gamma'] == 'scale' else (params['lambd'], params['gamma'])
        if key == self.fail_once:
            self.fail_once = None
            raise self.fail_error
        eligible = self.mode != 'invalid' or key == 'control'
        if key == 'control':
            score = .99
        elif self.mode == 'boundary':
            score = .5 + .1 * math.tanh(math.log(params['lambd']))
        else:
            score = max(.05, .9 - .04 * (math.log2(params['lambd'])**2 + math.log2(params['gamma'])**2))
        score = score if eligible else None
        folds = [{'fold': index, 'accuracy': score, 'eligible': eligible,
                  'status': 'trained' if eligible else 'single_class_selection'} for index in range(2)]
        return {'params': copy.deepcopy(params), 'mean_score': score,
                'eligible': eligible, 'folds': folds}

    @property
    def search_calls(self):
        return [p for p in self.calls if p['gamma'] != 'scale']


@contextmanager
def synthetic(fake):
    data = SimpleNamespace(X=np.arange(24, dtype=float).reshape(12, 2) / 255.,
                           y=np.array([2]*6+[3]*6), ids=np.array([f'train:{i}' for i in range(12)]),
                           digits=(2, 3), hashes={'images': 'synthetic', 'labels': 'synthetic'})
    with patch.object(_RUNNER, 'load_mnist_pair', return_value=data) as loader, \
            patch.object(_RUNNER, 'mnist_param_grid', return_value=grid()), \
            patch.object(_RUNNER, 'FoldEvaluator', side_effect=fake.factory), \
            patch('builtins.print'):
        yield loader, data


def test_dry_run_has_aligned_defaults_without_loading_or_fitting(capsys):
    with patch.object(_RUNNER, 'load_mnist_pair') as loader, patch.object(_RUNNER, 'FoldEvaluator') as evaluator:
        _RUNNER.main([])
    result = json.loads(capsys.readouterr().out)
    assert result['policy']['max_candidates'] == 500
    assert result['policy']['max_expansions'] == 3
    assert result['policy']['max_refinements'] is None
    assert result['policy']['stable_refinement_rounds'] == 2
    assert result['policy']['bracket_factor'] == 2.
    loader.assert_not_called()
    evaluator.assert_not_called()


def test_inadequate_initial_budget_rejected_before_data_or_output(tmp_path):
    args, fake = arguments(tmp_path, max_candidates=8), Scores()
    with synthetic(fake) as (loader, _), pytest.raises(ValueError, match='complete initial grid'):
        _RUNNER.run_search(args)
    loader.assert_not_called()
    assert not args.output_dir.exists() and not fake.calls


def test_repeated_refinement_earns_acceptance_and_locks_evidence(tmp_path):
    args, fake = arguments(tmp_path), Scores()
    with synthetic(fake) as (loader, data):
        result = _RUNNER.run_search(args)
    assert result['state'] == 'accepted' and not result['review_reasons']
    assert result['params']['lambd'] == result['params']['gamma'] == 1.
    assert result['mean_cv_accuracy'] == .9  # Control score .99 never competes.
    assert result['refinement_rounds'] >= 3 and result['stable_refinement_rounds'] >= 2
    assert all(item['within_factor'] and item['lower_valid'] and item['upper_valid']
               for item in result['winner_bracket'].values())
    assert result['final_evaluation_eligible']
    assert not result['final_refit_performed'] and not result['held_out_scored']
    assert not result['optimization_convergence_established']
    locked = _RUNNER.load_qualified_selection(args.output_dir)
    assert locked['params'] == result['params'] and locked['epochs'] == args.epochs
    assert locked['stopping'] == 'fixed'
    assert len(fake.search_calls) == result['attempted_candidates']
    loader.assert_called_once_with(args.data_dir, args.digits, partition='train')
    X, y, ids, folds, controls = fake.constructed[0]
    np.testing.assert_array_equal(X, data.X)
    np.testing.assert_array_equal(y, data.y)
    np.testing.assert_array_equal(ids, data.ids)
    assert 'validation_data' not in controls
    assert sorted(i for _, validation in folds for i in validation) == list(range(len(y)))
    for train, validation in folds:
        assert not set(train).intersection(validation)


@pytest.mark.parametrize('changes,mode,reason', [
    ({'max_candidates': 9}, 'peak', 'candidate_budget'),
    ({'max_expansions': 0}, 'boundary', 'expansion_budget'),
    ({'refinement_rounds': 1}, 'peak', 'refinement_budget'),
    ({}, 'invalid', 'ineligible'),
])
def test_unqualified_search_cannot_export_final_settings(tmp_path, changes, mode, reason):
    args, fake = arguments(tmp_path, **changes), Scores(mode)
    with synthetic(fake):
        result = _RUNNER.run_search(args)
    assert result['state'] == 'needs_review'
    assert any(reason in item for item in result['review_reasons'])
    assert not result['final_evaluation_eligible']
    assert not result['final_refit_performed'] and not result['held_out_scored']
    assert not (args.output_dir / 'qualified-selection.json').exists()
    with pytest.raises(ValueError, match='needs review'):
        _RUNNER.load_qualified_selection(args.output_dir)
    stages = read(args.output_dir / 'stages.json')
    assert all(stage['completed'] for stage in stages)  # No partial stage earns credit.
    assert len(records(args.output_dir)) == sum(len(stage['candidates']) for stage in stages)
    assert len(stages) == (2 if reason == 'refinement_budget' else 1)
    if mode == 'invalid':
        assert result['params'] is None and result['mean_cv_accuracy'] is None


@pytest.mark.parametrize('failure', [RuntimeError('synthetic interruption'), FloatingPointError('nonfinite fit'), KeyboardInterrupt()])
def test_failed_candidate_resume_reuses_committed_work_and_completed_run(tmp_path, failure):
    args, fake = arguments(tmp_path), Scores(fail_once=(1., 1.), fail_error=failure)
    with synthetic(fake):
        with pytest.raises(type(failure)):
            _RUNNER.run_search(args)
        saved = records(args.output_dir)
        assert saved and not read(args.output_dir / 'stages.json')[0]['completed']
        assert read(args.output_dir / 'status.json')['state'] in ('failed', 'interrupted')
        args.resume = True
        result = _RUNNER.run_search(args)
        calls, constructors = copy.deepcopy(fake.calls), len(fake.constructed)
        assert _RUNNER.run_search(args) == result
    assert len(fake.constructed) == constructors and fake.calls == calls
    assert records(args.output_dir)[:len(saved)] == saved
    assert result['state'] == 'accepted'
    assert len(fake.search_calls) == result['attempted_candidates'] + 1
    assert sum(p['gamma'] == 'scale' for p in fake.calls) == 1


def test_failed_control_is_not_committed_and_only_retried_on_resume(tmp_path):
    args, fake = arguments(tmp_path), Scores(fail_once='control')
    with synthetic(fake):
        with pytest.raises(RuntimeError, match='synthetic'):
            _RUNNER.run_search(args)
        assert not records(args.output_dir)
        assert not (args.output_dir / 'inherited-default-control.json').exists()
        args.resume = True
        result = _RUNNER.run_search(args)
    assert result['state'] == 'accepted'
    assert sum(p['gamma'] == 'scale' for p in fake.calls) == 2


def test_stage_completion_before_policy_cursor_is_counted_once_on_resume(tmp_path):
    args, fake = arguments(tmp_path), Scores()
    write = _RUNNER.write_json
    crashed = False

    def interrupt(path, value):
        nonlocal crashed
        if (not crashed and path.name == 'plan_state.json'
                and value['policy_state'].get('stable_rounds') == 1):
            crashed = True
            raise RuntimeError('before policy state commit')
        return write(path, value)

    with synthetic(fake):
        with patch.object(_RUNNER, 'write_json', side_effect=interrupt), pytest.raises(RuntimeError, match='policy state'):
            _RUNNER.run_search(args)
        before = records(args.output_dir)
        args.resume = True
        result = _RUNNER.run_search(args)
    assert result['state'] == 'accepted' and result['stable_refinement_rounds'] == 2
    assert result['refinement_rounds'] == 3
    assert records(args.output_dir)[:len(before)] == before
    assert len(fake.search_calls) == result['attempted_candidates']


@pytest.mark.parametrize('change', ['epochs', 'data_hashes', 'grid', 'splits', 'snapshot'])
def test_resume_rejects_identity_changes_without_fitting(tmp_path, change):
    args, fake = arguments(tmp_path), Scores(fail_once=(1., 1.))
    with synthetic(fake) as (_, data):
        with pytest.raises(RuntimeError):
            _RUNNER.run_search(args)
        before = (args.output_dir / 'records.jsonl').read_bytes()
        calls = copy.deepcopy(fake.calls)
        args.resume = True
        custom = None
        if change == 'epochs':
            args.epochs += 1
        elif change == 'data_hashes':
            data.hashes['images'] = 'different'
        elif change == 'grid':
            custom = grid() | {'lambd': [.25, 1., 5.]}
        elif change == 'splits':
            path = args.output_dir / 'splits.json'
            content = read(path)
            content['observation_ids'][0] = 'changed'
            path.write_text(json.dumps(content))
        else:
            (args.output_dir / 'implementation-snapshot/tune_mnist.py').write_text('changed')
        with pytest.raises(ValueError, match='[Rr]esume|[Ii]mplementation snapshot'):
            _RUNNER.run_search(args, param_grid=custom)
    assert fake.calls == calls and (args.output_dir / 'records.jsonl').read_bytes() == before


@pytest.mark.parametrize('suffix,valid', [(b'{"params":', True), (b'{"params":\n', False), (b'{"params":\n{}\n', False)])
def test_resume_repairs_only_truncated_final_record(tmp_path, suffix, valid):
    args, fake = arguments(tmp_path), Scores(fail_once=(1., 1.))
    with synthetic(fake):
        with pytest.raises(RuntimeError):
            _RUNNER.run_search(args)
        path = args.output_dir / 'records.jsonl'
        committed = records(args.output_dir)
        damaged = path.read_bytes() + suffix
        path.write_bytes(damaged)
        args.resume = True
        if valid:
            result = _RUNNER.run_search(args)
            assert result['state'] == 'accepted'
            assert records(args.output_dir)[:len(committed)] == committed
            assert (args.output_dir / 'records-interrupted.jsonl').read_bytes() == damaged
        else:
            calls = copy.deepcopy(fake.calls)
            with pytest.raises(ValueError):
                _RUNNER.run_search(args)
            assert path.read_bytes() == damaged and fake.calls == calls


def test_resume_preserves_complete_record_missing_final_newline(tmp_path):
    args, fake = arguments(tmp_path), Scores(fail_once=(1., 1.))
    with synthetic(fake):
        with pytest.raises(RuntimeError):
            _RUNNER.run_search(args)
        path = args.output_dir / 'records.jsonl'
        saved = records(args.output_dir)
        path.write_bytes(path.read_bytes().rstrip(b'\r\n'))
        args.resume = True
        assert _RUNNER.run_search(args)['state'] == 'accepted'
    assert records(args.output_dir)[:len(saved)] == saved


@pytest.mark.parametrize('name', ['config.json', 'splits.json', 'records.jsonl', 'selection.json', 'review.json'])
def test_qualified_settings_reject_changed_evidence(tmp_path, name):
    args, fake = arguments(tmp_path), Scores()
    with synthetic(fake):
        _RUNNER.run_search(args)
    path = args.output_dir / name
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='evidence differs'):
        _RUNNER.load_qualified_selection(args.output_dir)


@pytest.mark.parametrize('change', ['epochs', 'batch_size', 'stopping', 'random_state', 'preprocessing', 'package_version', 'missing_hashes'])
def test_qualified_settings_bind_controls_to_the_validated_configuration(tmp_path, change):
    args, fake = arguments(tmp_path), Scores()
    with synthetic(fake):
        _RUNNER.run_search(args)
    path = args.output_dir / 'qualified-selection.json'
    locked = read(path)
    if change == 'missing_hashes':
        locked['evidence_sha256'] = {}
    else:
        locked[change] = 'changed'
    path.write_text(json.dumps(locked))
    with pytest.raises(ValueError, match='controls differ|required evidence'):
        _RUNNER.load_qualified_selection(args.output_dir)
