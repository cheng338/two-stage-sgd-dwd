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
                  optimizer_backend='python', kernel_cache_bytes=0,
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
        self.execution_identity = _RUNNER.execution_identity(controls['optimizer_backend'])
        return SimpleNamespace(evaluate=self.evaluate)

    def evaluate(self, params, *, completed_folds=None, on_fold_completed=None):
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
        prefix = copy.deepcopy(completed_folds or [])
        folds[:len(prefix)] = prefix
        if on_fold_completed is not None:
            for fold in folds[len(prefix):]:
                on_fold_completed(copy.deepcopy(fold))
        return {'params': copy.deepcopy(params), 'mean_score': score,
                'eligible': eligible, 'folds': folds,
                'execution_identity': copy.deepcopy(self.execution_identity)}

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
    assert result['optimizer_backend'] == 'python'
    assert result['kernel_cache_bytes'] == 0
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
    assert locked['optimizer_backend'] == 'python' and locked['kernel_cache_bytes'] == 0
    assert locked['execution_identity']['extension_available'] is None
    assert len(fake.search_calls) == result['attempted_candidates']
    loader.assert_called_once_with(args.data_dir, args.digits, partition='train')
    X, y, ids, folds, controls = fake.constructed[0]
    np.testing.assert_array_equal(X, data.X)
    np.testing.assert_array_equal(y, data.y)
    np.testing.assert_array_equal(ids, data.ids)
    assert 'validation_data' not in controls
    assert controls['optimizer_backend'] == 'python' and controls['kernel_cache_bytes'] == 0
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


@pytest.mark.parametrize('change', ['epochs', 'data_hashes', 'grid', 'splits', 'snapshot',
                                  'optimizer_backend', 'kernel_cache_bytes'])
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
        elif change == 'optimizer_backend':
            args.optimizer_backend = 'auto'
        elif change == 'kernel_cache_bytes':
            args.kernel_cache_bytes = 1024
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


@pytest.mark.parametrize('change', ['epochs', 'batch_size', 'stopping', 'random_state', 'preprocessing',
                                  'package_version', 'optimizer_backend', 'kernel_cache_bytes',
                                  'execution_identity', 'missing_hashes'])
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


def test_cli_acceleration_options_are_explicit_and_keep_dry_run_data_free(capsys):
    with patch.object(_RUNNER, 'load_mnist_pair') as loader, patch.object(_RUNNER, 'FoldEvaluator') as evaluator:
        _RUNNER.main(['--optimizer-backend', 'native', '--kernel-cache-bytes', '67108864'])
    proposed = json.loads(capsys.readouterr().out)
    assert proposed['optimizer_backend'] == 'native'
    assert proposed['kernel_cache_bytes'] == 64 * 1024**2
    loader.assert_not_called()
    evaluator.assert_not_called()


@pytest.mark.parametrize('options', [['--optimizer-backend', 'unknown'], ['--kernel-cache-bytes', '-1']])
def test_cli_invalid_acceleration_rejected_before_data(options):
    with patch.object(_RUNNER, 'load_mnist_pair') as loader, pytest.raises(SystemExit):
        _RUNNER.main(options)
    loader.assert_not_called()


@contextmanager
def native_identity_fixture(path, available=True):
    """Synthetic bytes exercise snapshot identity without importing a binary."""
    package = path / 'package'
    package.mkdir()
    init = package / '__init__.py'
    init.write_text('# synthetic package source\n', encoding='utf-8')
    binary = package / '_sgd_accel.pyd'
    binary.write_bytes(b'synthetic native build 1')
    availability = [available]

    def identity(backend):
        return {'requested_backend': backend, 'extension_available': availability[0],
                'extension_filename': binary.name if availability[0] else None,
                'extension_sha256': hashlib.sha256(binary.read_bytes()).hexdigest() if availability[0] else None,
                'package_version': _RUNNER.two_stage_sgd_dwd.__version__}

    with patch.object(_RUNNER.two_stage_sgd_dwd, '__file__', str(init)), \
            patch.object(_RUNNER, 'execution_identity', side_effect=identity):
        yield binary, availability


def test_native_configuration_snapshots_binary_and_locks_portable_identity(tmp_path):
    args, fake = arguments(tmp_path, optimizer_backend='native', kernel_cache_bytes=67108864), Scores()
    with native_identity_fixture(tmp_path) as (binary, _), synthetic(fake):
        _RUNNER.run_search(args)
        config = read(args.output_dir / 'config.json')
        locked = _RUNNER.load_qualified_selection(args.output_dir)
        assert (args.output_dir / 'implementation-snapshot' / binary.name).read_bytes() == binary.read_bytes()
        expected = hashlib.sha256(binary.read_bytes()).hexdigest()
        assert config['implementation_sha256'][binary.name] == expected
        assert config['execution_identity']['extension_sha256'] == expected
        assert str(tmp_path) not in json.dumps(config['execution_identity'])
        assert locked['execution_identity'] == config['execution_identity']
        assert locked['kernel_cache_bytes'] == 67108864
        assert fake.constructed[0][-1]['optimizer_backend'] == 'native'
        assert fake.constructed[0][-1]['kernel_cache_bytes'] == 67108864


@pytest.mark.parametrize('change', ['binary', 'availability', 'snapshot'])
def test_native_resume_rejects_binary_availability_or_saved_snapshot_changes(tmp_path, change):
    args, fake = arguments(tmp_path, optimizer_backend='auto'), Scores(fail_once=(1., 1.))
    with native_identity_fixture(tmp_path) as (binary, available), synthetic(fake):
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            _RUNNER.run_search(args)
        before = (args.output_dir / 'records.jsonl').read_bytes()
        calls = copy.deepcopy(fake.calls)
        if change == 'binary':
            binary.write_bytes(b'synthetic native build 2')
        elif change == 'availability':
            available[0] = False
        else:
            (args.output_dir / 'implementation-snapshot' / binary.name).write_bytes(b'changed snapshot')
        args.resume = True
        with pytest.raises(ValueError, match='Resume|snapshot'):
            _RUNNER.run_search(args)
        assert fake.calls == calls
        assert (args.output_dir / 'records.jsonl').read_bytes() == before


def test_auto_without_extension_is_recorded_and_new_availability_blocks_resume(tmp_path):
    args, fake = arguments(tmp_path, optimizer_backend='auto'), Scores()
    with native_identity_fixture(tmp_path, available=False) as (binary, available), synthetic(fake):
        _RUNNER.run_search(args)
        config = read(args.output_dir / 'config.json')
        assert config['execution_identity']['extension_available'] is False
        assert binary.name not in config['implementation_sha256']
        assert not (args.output_dir / 'implementation-snapshot' / binary.name).exists()
        _RUNNER.load_qualified_selection(args.output_dir)
        available[0] = True
        calls = copy.deepcopy(fake.calls)
        args.resume = True
        with pytest.raises(ValueError, match='Resume'):
            _RUNNER.run_search(args)
        with pytest.raises(ValueError, match='identity differs'):
            _RUNNER.load_qualified_selection(args.output_dir)
        assert fake.calls == calls


def test_qualified_native_settings_reject_live_binary_swap(tmp_path):
    args, fake = arguments(tmp_path, optimizer_backend='native'), Scores()
    with native_identity_fixture(tmp_path) as (binary, _), synthetic(fake):
        _RUNNER.run_search(args)
        binary.write_bytes(b'synthetic native build 2')
        with pytest.raises(ValueError, match='identity differs'):
            _RUNNER.load_qualified_selection(args.output_dir)


def test_explicit_native_unavailable_fails_before_data_and_output(tmp_path):
    args, fake = arguments(tmp_path, optimizer_backend='native'), Scores()
    with synthetic(fake) as (loader, _), \
            patch.object(_RUNNER, 'execution_identity', side_effect=ImportError('native extension unavailable')):
        with pytest.raises(ImportError, match='native extension unavailable'):
            _RUNNER.run_search(args)
    loader.assert_not_called()
    assert not args.output_dir.exists() and not fake.calls


def test_evaluator_identity_disagreement_is_not_committed(tmp_path):
    args, fake = arguments(tmp_path), Scores()
    evaluate = fake.evaluate

    def mismatched(params, **kwargs):
        record = evaluate(params, **kwargs)
        record['execution_identity']['package_version'] = 'different'
        return record

    with synthetic(fake), patch.object(fake, 'evaluate', side_effect=mismatched):
        with pytest.raises(ValueError, match='Evaluator execution identity'):
            _RUNNER.run_search(args)
    assert not records(args.output_dir)
    assert not (args.output_dir / 'inherited-default-control.json').exists()
    assert read(args.output_dir / 'status.json')['state'] == 'failed'


def test_exact_adjacent_candidates_survive_interruption_and_resume(tmp_path):
    adjacent = math.nextafter(1., math.inf)
    fractions = [math.nextafter(.3, 0.), .3]
    custom = {'kernel': ['rbf'], 'lambd': [1., adjacent], 'gamma': [1., adjacent],
              'retention_fraction': fractions}
    args, fake = arguments(tmp_path, max_candidates=8), Scores()
    original = fake.evaluate
    interrupted = False

    def evaluate(params, **kwargs):
        nonlocal interrupted
        result = original(params, **kwargs)
        if (params['gamma'] == 1. and params['lambd'] == 1.
                and params['retention_fraction'] == fractions[1] and not interrupted):
            interrupted = True
            raise RuntimeError('adjacent candidate interruption')
        if params['gamma'] != 'scale':
            from two_stage_sgd_dwd._core import select_near_boundary
            chosen = select_near_boundary(np.arange(10.), params['retention_fraction'],
                                          np.random.default_rng(17))
            result['n_retained_on_ten_rows'] = len(chosen)
        return result

    with synthetic(fake), patch.object(fake, 'evaluate', side_effect=evaluate):
        with pytest.raises(RuntimeError, match='adjacent candidate'):
            _RUNNER.run_search(args, param_grid=custom)
        saved = records(args.output_dir)
        assert len(saved) == 1 and saved[0]['params']['retention_fraction'] == fractions[0]
        args.resume = True
        result = _RUNNER.run_search(args, param_grid=custom)
        final = records(args.output_dir)
        calls = copy.deepcopy(fake.calls)
        assert _RUNNER.run_search(args, param_grid=custom) == result
        assert fake.calls == calls
    assert len(final) == 8 and len(fake.search_calls) == 9  # One interrupted attempt.
    assert final[:len(saved)] == saved
    coordinates = {(r['params']['lambd'], r['params']['gamma'],
                    r['params']['retention_fraction']) for r in final}
    assert coordinates == {(l, g, f) for l in (1., adjacent) for g in (1., adjacent)
                           for f in fractions}
    assert all(r['n_retained_on_ten_rows'] == (3 if r['params']['retention_fraction'] == .3 else 2)
               for r in final)
    assert result['attempted_candidates'] == 8
    assert read(args.output_dir / 'config.json')['grid'] == custom
    stages = read(args.output_dir / 'stages.json')
    assert len(stages) == 1 and stages[0]['completed']
    assert {tuple(p[name] for name in ('lambd', 'gamma', 'retention_fraction'))
            for p in stages[0]['candidates']} == coordinates


def _output_bytes(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob('*') if path.is_file()}


@pytest.mark.parametrize('completed', [False, True])
@pytest.mark.parametrize('legacy', ['config', 'plan', 'policy', 'missing_policy_schema'])
def test_legacy_grid_identity_rejected_before_evaluation_or_output_changes(tmp_path, completed, legacy):
    args = arguments(tmp_path)
    fake = Scores(fail_once=None if completed else (1., 1.))
    with synthetic(fake):
        if completed:
            _RUNNER.run_search(args)
        else:
            with pytest.raises(RuntimeError, match='synthetic interruption'):
                _RUNNER.run_search(args)
        filename = 'config.json' if legacy == 'config' else 'plan_state.json'
        path = args.output_dir / filename
        value = read(path)
        if legacy == 'config':
            value['schema_version'] = 3
        elif legacy == 'plan':
            value['schema_version'] = 2
        elif legacy == 'policy':
            value['policy_state']['schema_version'] = 1
        else:
            value['policy_state'].pop('schema_version')
        path.write_text(json.dumps(value), encoding='utf-8')
        before = _output_bytes(args.output_dir)
        calls, constructors = copy.deepcopy(fake.calls), len(fake.constructed)
        args.resume = True
        with pytest.raises(ValueError, match='fresh output directory'):
            _RUNNER.run_search(args)
        assert fake.calls == calls and len(fake.constructed) == constructors
        assert _output_bytes(args.output_dir) == before


def test_legacy_qualified_selection_requires_fresh_output_directory(tmp_path):
    args, fake = arguments(tmp_path), Scores()
    with synthetic(fake):
        _RUNNER.run_search(args)
    config_path = args.output_dir / 'config.json'
    config = read(config_path)
    config['schema_version'] = 3
    config_path.write_text(json.dumps(config), encoding='utf-8')
    lock_path = args.output_dir / 'qualified-selection.json'
    locked = read(lock_path)
    locked['evidence_sha256']['config.json'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    lock_path.write_text(json.dumps(locked), encoding='utf-8')
    before = _output_bytes(args.output_dir)
    with pytest.raises(ValueError, match='fresh output directory'):
        _RUNNER.load_qualified_selection(args.output_dir)
    assert _output_bytes(args.output_dir) == before
