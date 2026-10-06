"""Explicit numerical-build migrations preserve completed CV evidence."""
from contextlib import contextmanager
import copy
import json
from unittest.mock import patch

import pytest
from threadpoolctl import threadpool_limits

from test_tune_mnist import _RUNNER, Scores, arguments, read, records, synthetic


def output_bytes(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob('*') if path.is_file() and path.name != '.run.lock'}


@pytest.fixture
def runtimes(monkeypatch):
    with threadpool_limits(1):
        initial = _RUNNER.execution_identity('python')
    variants = {'a': initial}
    for name, api in [('b', 'openblas'), ('c', 'mkl')]:
        identity = copy.deepcopy(initial)
        pool = next(item for item in identity['numerical_runtime']['threadpools']
                    if item['user_api'] == 'blas')
        pool.update(internal_api=api, prefix='test_' + api,
                    filename='test_' + api + '.dll', version='test-version',
                    sha256=name * 64)
        variants[name] = identity
    active = {'name': 'a', 'variants': variants}

    def identity(backend):
        value = copy.deepcopy(variants[active['name']])
        value['requested_backend'] = backend
        return value

    monkeypatch.setattr(_RUNNER, 'execution_identity', identity)
    return active


@contextmanager
def started(path, runtimes, *, fail_once=(4., 4.), completed=False):
    args = arguments(path, allow_blas_change=False)
    fake = Scores(fail_once=None if completed else fail_once)
    with synthetic(fake):
        if completed:
            _RUNNER.run_search(args)
        else:
            with pytest.raises(RuntimeError, match='synthetic interruption'):
                _RUNNER.run_search(args)
        yield args, fake


def reject_without_changes(args, fake, *, loader=False, allow=False):
    before = output_bytes(args.output_dir)
    calls, constructors = copy.deepcopy(fake.calls), len(fake.constructed)
    with pytest.raises(ValueError):
        if loader:
            _RUNNER.load_qualified_selection(args.output_dir, allow_blas_change=allow)
        else:
            args.resume = True
            args.allow_blas_change = allow
            _RUNNER.run_search(args)
    assert output_bytes(args.output_dir) == before
    assert fake.calls == calls and len(fake.constructed) == constructors


def test_default_rejects_new_runtime_before_fitting_or_recovery(tmp_path, runtimes):
    with started(tmp_path, runtimes) as (args, fake):
        path = args.output_dir / 'records.jsonl'
        path.write_bytes(path.read_bytes() + b'{"params":')
        runtimes['name'] = 'b'
        reject_without_changes(args, fake)
        assert not (args.output_dir / 'records-interrupted.jsonl').exists()


def test_opt_in_preserves_config_receipts_and_completed_candidates(tmp_path, runtimes):
    with started(tmp_path, runtimes) as (args, fake):
        saved = records(args.output_dir)
        prefix = (args.output_dir / 'records.jsonl').read_bytes()
        config = (args.output_dir / 'config.json').read_bytes()
        assert len(saved) == 8
        runtimes['name'] = 'b'
        args.resume, args.allow_blas_change = True, True
        result = _RUNNER.run_search(args)
        rows = records(args.output_dir)
        assert result['state'] == 'accepted' and result['mean_cv_accuracy'] == .9
        assert result['mixed_runtime'] and result['timings_require_same_runtime']
        assert rows[:8] == saved
        assert (args.output_dir / 'records.jsonl').read_bytes().startswith(prefix)
        assert (args.output_dir / 'config.json').read_bytes() == config
        assert all('runtime_transition' not in row for row in saved)
        transitions = [(i, row['runtime_transition']) for i, row in enumerate(rows)
                       if 'runtime_transition' in row]
        assert len(transitions) == 1 and transitions[0][0] == 8
        expected = {'schema_version': 1, 'reason': 'explicit_blas_change',
                    'from_identity_sha256': _RUNNER._content_sha256(runtimes['variants']['a']),
                    'to_identity_sha256': _RUNNER._content_sha256(runtimes['variants']['b'])}
        assert transitions[0][1] == expected
        assert result['runtime_history'] == [
            {'candidate_sequence': None, 'execution_identity': runtimes['variants']['a']},
            {'candidate_sequence': 8, 'execution_identity': runtimes['variants']['b']}]
        assert all(row['execution_identity'] == runtimes['variants']['b'] for row in rows[8:])
        # The winner was already committed in the original environment.
        assert result['selected_execution_identity'] == runtimes['variants']['a']
        old_keys = {_RUNNER.candidate_key(row['params']) for row in saved}
        assert all(sum(_RUNNER.candidate_key(p) == key for p in fake.search_calls) == 1
                   for key in old_keys)
        before = output_bytes(args.output_dir)
        locked = _RUNNER.load_qualified_selection(args.output_dir, allow_blas_change=True)
        assert locked['consumer_execution_identity'] == runtimes['variants']['b']
        assert locked['selected_execution_identity'] == runtimes['variants']['a']
        assert locked['consumer_runtime_differs_from_selected']
        assert output_bytes(args.output_dir) == before


@pytest.mark.parametrize('runtime', ['a', 'b', 'c'])
@pytest.mark.parametrize('loader', [False, True])
def test_mixed_history_requires_explicit_opt_in_on_every_read(tmp_path, runtimes, runtime, loader):
    with started(tmp_path, runtimes) as (args, fake):
        runtimes['name'] = 'b'
        args.resume, args.allow_blas_change = True, True
        _RUNNER.run_search(args)
        runtimes['name'] = runtime
        reject_without_changes(args, fake, loader=loader)


def test_a_to_b_to_a_records_both_boundaries_without_rewriting_history(tmp_path, runtimes):
    with started(tmp_path, runtimes, fail_once=(1., 1.)) as (args, fake):
        first = records(args.output_dir)
        assert len(first) == 4
        runtimes['name'] = 'b'
        args.resume, args.allow_blas_change = True, True
        fake.fail_once = (4., 4.)
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            _RUNNER.run_search(args)
        second = records(args.output_dir)
        assert len(second) == 8 and second[:4] == first
        runtimes['name'] = 'a'
        result = _RUNNER.run_search(args)
        assert result['state'] == 'accepted' and result['mixed_runtime']
        final = records(args.output_dir)
        assert final[:8] == second
        assert [i for i, row in enumerate(final) if 'runtime_transition' in row] == [4, 8]
        assert [item['execution_identity'] for item in result['runtime_history']] == [
            runtimes['variants'][name] for name in ('a', 'b', 'a')]
        reject_without_changes(args, fake)


@pytest.mark.parametrize('completed', [False, True])
def test_opt_in_same_runtime_does_not_invent_migration(tmp_path, runtimes, completed):
    with started(tmp_path, runtimes, completed=completed) as (args, fake):
        args.resume, args.allow_blas_change = True, True
        result = _RUNNER.run_search(args)
        assert not result['mixed_runtime']
        assert all('runtime_transition' not in row for row in records(args.output_dir))
        assert len(result['runtime_history']) == 1


@pytest.mark.parametrize('loader', [False, True])
def test_read_only_cross_runtime_consumption_does_not_invent_completed_work(tmp_path, runtimes, loader):
    with started(tmp_path, runtimes, completed=True) as (args, fake):
        before = output_bytes(args.output_dir)
        calls, constructors = copy.deepcopy(fake.calls), len(fake.constructed)
        runtimes['name'] = 'b'
        reject_without_changes(args, fake, loader=loader)
        if loader:
            result = _RUNNER.load_qualified_selection(args.output_dir, allow_blas_change=True)
        else:
            args.resume, args.allow_blas_change = True, True
            result = _RUNNER.run_search(args)
        assert not result['mixed_runtime'] and len(result['runtime_history']) == 1
        assert result['consumer_execution_identity'] == runtimes['variants']['b']
        assert result['consumer_runtime_differs_from_selected']
        assert output_bytes(args.output_dir) == before
        assert fake.calls == calls and len(fake.constructed) == constructors


@pytest.mark.parametrize('field', ['python', 'numpy_version', 'scipy_version', 'sklearn_version',
                                  'threads', 'native_binary', 'backend'])
def test_opt_in_does_not_accept_unrelated_runtime_changes(tmp_path, runtimes, field):
    with started(tmp_path, runtimes) as (args, fake):
        runtimes['name'] = 'b'
        target = runtimes['variants']['b']
        runtime = target['numerical_runtime']
        if field == 'python':
            runtime['python'] = '3.99.0'
        elif field.endswith('_version'):
            runtime['packages'][field.removesuffix('_version')]['version'] = '99.0.0'
        elif field == 'threads':
            next(p for p in runtime['threadpools'] if p['user_api'] == 'blas')['num_threads'] = 2
        elif field == 'native_binary':
            target.update(extension_available=True, extension_filename='different.pyd',
                          extension_sha256='e' * 64)
        else:
            args.optimizer_backend = 'auto'
        reject_without_changes(args, fake, allow=True)


@pytest.mark.parametrize('field', ['epochs', 'random_state', 'threads', 'kernel_cache_bytes'])
def test_opt_in_keeps_original_fit_and_execution_controls(tmp_path, runtimes, field):
    with started(tmp_path, runtimes) as (args, fake):
        runtimes['name'] = 'b'
        setattr(args, field, getattr(args, field) + 1)
        reject_without_changes(args, fake, allow=True)


def test_failed_first_target_candidate_does_not_commit_transition(tmp_path, runtimes):
    with started(tmp_path, runtimes) as (args, fake):
        prefix = (args.output_dir / 'records.jsonl').read_bytes()
        runtimes['name'] = 'b'
        fake.fail_once = (4., 4.)
        args.resume, args.allow_blas_change = True, True
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            _RUNNER.run_search(args)
        assert (args.output_dir / 'records.jsonl').read_bytes() == prefix
        assert all('runtime_transition' not in row for row in records(args.output_dir))
        runtimes['name'] = 'a'
        args.allow_blas_change = False
        assert _RUNNER.run_search(args)['state'] == 'accepted'
        assert all('runtime_transition' not in row for row in records(args.output_dir))


def test_evaluator_cannot_supply_reserved_transition_metadata(tmp_path, runtimes):
    args, fake = arguments(tmp_path, allow_blas_change=True), Scores()
    evaluate = fake.evaluate

    def forged(params):
        row = evaluate(params)
        if params['gamma'] != 'scale':
            row['runtime_transition'] = {'reason': 'forged'}
        return row

    with synthetic(fake), patch.object(fake, 'evaluate', side_effect=forged):
        with pytest.raises(ValueError):
            _RUNNER.run_search(args)
    assert records(args.output_dir) == []


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_programmatic_opt_in_requires_boolean(tmp_path, runtimes, value):
    args, fake = arguments(tmp_path, allow_blas_change=value), Scores()
    with synthetic(fake):
        with pytest.raises((TypeError, ValueError)):
            _RUNNER.run_search(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize('kind', ['grid', 'data', 'splits', 'source', 'snapshot'])
def test_opt_in_keeps_data_folds_grid_and_implementation_fixed(tmp_path, runtimes, kind):
    with started(tmp_path, runtimes) as (args, fake):
        runtimes['name'] = 'b'
        if kind == 'grid':
            changed = copy.deepcopy(_RUNNER.mnist_param_grid.return_value)
            changed['lambd'] = [.5, 1., 4.]
            _RUNNER.mnist_param_grid.return_value = changed
        elif kind == 'data':
            _RUNNER.load_mnist_pair.return_value.hashes['images'] = 'different-data'
        elif kind == 'splits':
            path = args.output_dir / 'splits.json'
            value = read(path)
            value['observation_ids'].reverse()
            path.write_text(json.dumps(value), encoding='utf-8')
        elif kind == 'snapshot':
            path = args.output_dir / 'implementation-snapshot' / '_mnist_cv.py'
            path.write_bytes(path.read_bytes() + b'\n# changed snapshot\n')
        else:
            implementation = _RUNNER._implementation_bytes

            def changed_source(identity):
                value = implementation(identity)
                value['_mnist_cv.py'] += b'\n# changed source\n'
                return value

            with patch.object(_RUNNER, '_implementation_bytes', side_effect=changed_source):
                reject_without_changes(args, fake, allow=True)
            return
        reject_without_changes(args, fake, allow=True)


def reseal(directory, rows):
    root = _RUNNER._run_sha256(read(directory / 'config.json'), read(directory / 'splits.json'))
    previous = root
    for sequence, row in enumerate(rows):
        row.pop('_receipt')
        rows[sequence] = _RUNNER._seal_record(row, run_sha256=root, sequence=sequence,
                                            previous_sha256=previous)
        previous = rows[sequence]['_receipt']['sha256']
    (directory / 'records.jsonl').write_text(
        ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')


@pytest.mark.parametrize('kind', ['missing', 'wrong_origin', 'wrong_target', 'boolean_schema',
                                 'extra_field', 'redundant', 'incompatible_version'])
def test_valid_checksums_do_not_excuse_invalid_transition_history(tmp_path, runtimes, kind):
    with started(tmp_path, runtimes, fail_once=(1., 1.)) as (args, fake):
        runtimes['name'] = 'b'
        args.resume, args.allow_blas_change = True, True
        fake.fail_once = (4., 4.)
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            _RUNNER.run_search(args)
        rows = records(args.output_dir)
        assert len(rows) == 8
        transition = rows[4]['runtime_transition']
        if kind == 'missing':
            del rows[4]['runtime_transition']
        elif kind == 'wrong_origin':
            transition['from_identity_sha256'] = 'f' * 64
        elif kind == 'wrong_target':
            transition['to_identity_sha256'] = 'f' * 64
        elif kind == 'boolean_schema':
            transition['schema_version'] = True
        elif kind == 'extra_field':
            transition['unexpected'] = 1
        elif kind == 'redundant':
            rows[5]['runtime_transition'] = _RUNNER._runtime_transition(
                rows[4]['execution_identity'], rows[5]['execution_identity'])
        else:
            for row in rows[4:]:
                row['execution_identity']['numerical_runtime']['packages']['numpy']['version'] = '99.0.0'
            rows[4]['runtime_transition'] = _RUNNER._runtime_transition(
                rows[3]['execution_identity'], rows[4]['execution_identity'])
        reseal(args.output_dir, rows)
        reject_without_changes(args, fake, allow=True)


def test_pinned_vendor_builds_can_change_scientific_hashes(runtimes):
    initial, target = runtimes['variants']['a'], runtimes['variants']['b']
    for item in target['numerical_runtime']['packages'].values():
        item['sha256'] = 'e' * 64
    for item in target['numerical_runtime']['binaries']:
        item['sha256'] = 'e' * 64
    _RUNNER._check_blas_change(initial, target, threads=1)


@pytest.mark.parametrize('kind', ['unchanged_pools', 'module_removed', 'abi_filename',
                                 'platform', 'unknown_api', 'boolean_threads'])
def test_compatibility_requires_matching_inventory_platform_and_known_pools(runtimes, kind):
    initial, target = runtimes['variants']['a'], runtimes['variants']['b']
    runtime = target['numerical_runtime']
    if kind == 'unchanged_pools':
        runtime['threadpools'] = copy.deepcopy(initial['numerical_runtime']['threadpools'])
        runtime['packages']['numpy']['sha256'] = 'f' * 64
    elif kind == 'module_removed':
        runtime['binaries'].pop()
    elif kind == 'abi_filename':
        runtime['binaries'][0]['filename'] = 'another-python-abi.pyd'
    elif kind == 'platform':
        runtime['system'] = 'OtherOS'
    elif kind == 'unknown_api':
        next(p for p in runtime['threadpools'] if p['user_api'] == 'blas')['internal_api'] = 'unknown'
    else:
        runtime['threadpools'][0]['num_threads'] = True
    with pytest.raises(ValueError):
        _RUNNER._check_blas_change(initial, target, threads=1)


@pytest.mark.parametrize('dependency', ['joblib', 'threadpoolctl'])
@pytest.mark.parametrize('loader', [False, True])
def test_opt_in_does_not_bypass_supporting_dependency_versions(tmp_path, runtimes, dependency, loader):
    with started(tmp_path, runtimes, completed=loader) as (args, fake):
        runtimes['name'] = 'b'
        with patch.object(getattr(_RUNNER, dependency), '__version__', '99.0.0'):
            reject_without_changes(args, fake, loader=loader, allow=True)


@pytest.mark.parametrize('enabled', [False, True])
def test_cli_passes_explicit_migration_choice(enabled):
    argv = ['--run', '--resume', '--data-dir', 'input', '--output-dir', 'output']
    if enabled:
        argv.append('--allow-blas-change')
    with patch.object(_RUNNER, 'run_search', return_value='accepted') as run:
        assert _RUNNER.main(argv) == 'accepted'
    assert run.call_args.args[0].allow_blas_change is enabled
