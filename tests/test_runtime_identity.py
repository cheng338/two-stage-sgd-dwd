"""CV evidence binds the numerical runtime, including effective thread limits."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from test_mnist_cv import mnist_cv, small_data
from test_tune_mnist import _RUNNER, Scores, arguments, read, records, synthetic

_RUNTIME = sys.modules[mnist_cv.runtime_identity.__module__]


def output_bytes(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob('*') if path.is_file() and path.name != '.run.lock'}


def pool_threads():
    return sorted((pool['internal_api'], pool['prefix'], pool['num_threads'])
                  for pool in threadpool_info())


@pytest.fixture
def fake_runtime(monkeypatch):
    identity = {'schema_version': 1, 'packages': {'numpy': 'same-version'},
                'libraries': [{'internal_api': 'blas-a', 'version': '1',
                               'sha256': 'a' * 64, 'num_threads': 1}]}
    monkeypatch.setitem(_RUNNER.execution_identity.__globals__, 'runtime_identity',
                        lambda: copy.deepcopy(identity))
    return identity


def test_runtime_identity_is_stable_portable_json_and_returned_by_value():
    with threadpool_limits(1):
        first = mnist_cv.runtime_identity()
        saved = copy.deepcopy(first)
        second = mnist_cv.runtime_identity()
        assert first == second
        first.clear()
        assert mnist_cv.runtime_identity() == saved
    serialized = json.dumps(second, allow_nan=False, sort_keys=True)
    assert str(Path(mnist_cv.np.__file__).parent) not in serialized
    assert second['schema_version'] == 1
    assert set(second['packages']) == {'numpy', 'scipy', 'sklearn'}
    assert second['binaries'] and second['threadpools']
    for item in [*second['binaries'], *second['threadpools'], *second['packages'].values()]:
        assert len(item['sha256']) == 64 and int(item['sha256'], 16) >= 0
        assert Path(item['filename']).name == item['filename']


@pytest.fixture
def controlled_runtime(tmp_path, monkeypatch):
    binary = tmp_path / 'scientific.pyd'
    library = tmp_path / 'blas.dll'
    binary.write_bytes(b'scientific binary v1')
    library.write_bytes(b'blas library v1')
    pools = [{'filepath': str(library), 'user_api': 'blas', 'internal_api': 'mockblas',
              'prefix': 'testblas', 'version': '1', 'num_threads': 1}]
    monkeypatch.setattr(_RUNTIME, '_loaded_binaries', lambda: {'numpy.test': binary})
    monkeypatch.setattr(_RUNTIME, 'threadpool_info', lambda: copy.deepcopy(pools))
    return binary, library, pools


@pytest.mark.parametrize('field,value', [('version', '2'), ('num_threads', 2),
                                          ('internal_api', 'otherblas')])
def test_actual_pool_metadata_changes_runtime_identity(controlled_runtime, field, value):
    _, _, pools = controlled_runtime
    original = _RUNTIME.runtime_identity()
    pools[0][field] = value
    assert _RUNTIME.runtime_identity() != original


@pytest.mark.parametrize('target', [0, 1])
def test_changed_binary_bytes_invalidate_cached_identity(controlled_runtime, target):
    first = _RUNTIME.runtime_identity()
    path = controlled_runtime[target]
    previous = path.stat()
    path.write_bytes(b'changed library contents')
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 10_000_000))
    second = _RUNTIME.runtime_identity()
    assert second != first
    group = 'binaries' if target == 0 else 'threadpools'
    assert second[group][0]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_unchanged_files_reuse_hashes_without_reading_bytes(controlled_runtime, monkeypatch):
    first = _RUNTIME.runtime_identity()

    def unexpected(*args, **kwargs):
        raise AssertionError('Unchanged numerical files should reuse cached hashes')

    monkeypatch.setattr(Path, 'open', unexpected)
    assert _RUNTIME.runtime_identity() == first


def test_same_size_rewrite_with_restored_mtime_still_invalidates_hash(tmp_path):
    path = tmp_path / 'runtime.dll'
    path.write_bytes(b'old bytes')
    original = _RUNTIME._file_sha256(path)
    previous = path.stat()
    # A replacement has a new file identity even if length and mtime agree.
    replacement = tmp_path / 'replacement.dll'
    replacement.write_bytes(b'new bytes')
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_size == previous.st_size
    assert path.stat().st_mtime_ns == previous.st_mtime_ns
    assert _RUNTIME._file_sha256(path) != original


def test_file_changed_during_hash_is_rejected_and_not_cached(tmp_path, monkeypatch):
    path = tmp_path / 'runtime.dll'
    path.write_bytes(b'original bytes')
    original_stamp = _RUNTIME._stamp
    calls = []

    def changed_stamp(candidate):
        value = original_stamp(candidate)
        calls.append(candidate)
        if len(calls) == 2:
            return (*value[:-1], value[-1] + 1)
        return value

    monkeypatch.setattr(_RUNTIME, '_stamp', changed_stamp)
    with pytest.raises(RuntimeError, match='changed while'):
        _RUNTIME._file_sha256(path)
    assert path.resolve() not in _RUNTIME._HASH_CACHE


@pytest.mark.parametrize('failure', ['missing', 'unreadable', 'no_blas'])
def test_unidentifiable_runtime_fails_closed(controlled_runtime, monkeypatch, failure):
    _, library, pools = controlled_runtime
    if failure == 'missing':
        library.unlink()
        expected = FileNotFoundError
    elif failure == 'unreadable':
        original_open = Path.open

        def denied(path, *args, **kwargs):
            if path == library:
                raise PermissionError('runtime library unreadable')
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'open', denied)
        expected = PermissionError
    else:
        pools.clear()
        expected = RuntimeError
    with pytest.raises(expected):
        _RUNTIME.runtime_identity()


def test_runtime_identity_is_portable_when_equal_libraries_move(controlled_runtime, tmp_path, monkeypatch):
    binary, library, pools = controlled_runtime
    first = _RUNTIME.runtime_identity()
    moved = tmp_path / 'different-environment'
    moved.mkdir()
    moved_binary, moved_library = moved / binary.name, moved / library.name
    moved_binary.write_bytes(binary.read_bytes())
    moved_library.write_bytes(library.read_bytes())
    monkeypatch.setattr(_RUNTIME, '_loaded_binaries', lambda: {'numpy.test': moved_binary})
    pools[0]['filepath'] = str(moved_library)
    assert _RUNTIME.runtime_identity() == first


def test_pool_enumeration_order_does_not_change_identity(controlled_runtime):
    _, library, pools = controlled_runtime
    pools.append({'filepath': str(library), 'user_api': 'openmp', 'internal_api': 'openmp',
                  'prefix': 'testomp', 'version': None, 'num_threads': 1})
    first = _RUNTIME.runtime_identity()
    pools.reverse()
    assert _RUNTIME.runtime_identity() == first


def test_runtime_identity_distinguishes_effective_threads_and_restores_them():
    ambient = pool_threads()
    with threadpool_limits(1):
        single = mnist_cv.runtime_identity()
        with threadpool_limits(2):
            assert any(item[-1] == 2 for item in pool_threads())
            double = mnist_cv.runtime_identity()
        assert mnist_cv.runtime_identity() == single
    assert single != double
    assert pool_threads() == ambient


@pytest.mark.parametrize('field,value', [
    ('internal_api', 'blas-b'), ('version', '2'), ('sha256', 'b' * 64),
    ('num_threads', 2),
])
@pytest.mark.parametrize('completed', [False, True])
def test_runtime_change_rejects_resume_without_refitting_or_rewriting(
        tmp_path, fake_runtime, field, value, completed):
    args = arguments(tmp_path)
    fake = Scores(fail_once=None if completed else (1., 1.))
    with synthetic(fake):
        if completed:
            _RUNNER.run_search(args)
        else:
            with pytest.raises(RuntimeError, match='synthetic interruption'):
                _RUNNER.run_search(args)
        assert read(args.output_dir / 'config.json')['execution_identity']['numerical_runtime'] == fake_runtime
        before = output_bytes(args.output_dir)
        calls, constructions = copy.deepcopy(fake.calls), len(fake.constructed)
        fake_runtime['libraries'][0][field] = value
        args.resume = True
        with pytest.raises(ValueError, match='[Rr]esume|[Ee]xecution identity|[Rr]untime'):
            _RUNNER.run_search(args)
    assert fake.calls == calls and len(fake.constructed) == constructions
    assert output_bytes(args.output_dir) == before


def test_same_runtime_resume_reuses_all_committed_records(tmp_path, fake_runtime):
    args, fake = arguments(tmp_path), Scores(fail_once=(1., 1.))
    with synthetic(fake):
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            _RUNNER.run_search(args)
        saved = records(args.output_dir)
        args.resume = True
        result = _RUNNER.run_search(args)
        calls = copy.deepcopy(fake.calls)
        assert _RUNNER.run_search(args) == result
        assert _RUNNER.load_qualified_selection(args.output_dir)['params'] == result['params']
    assert records(args.output_dir)[:len(saved)] == saved
    assert fake.calls == calls
    assert all(row['execution_identity']['numerical_runtime'] == fake_runtime
               for row in records(args.output_dir))
    snapshot = args.output_dir / 'implementation-snapshot' / '_runtime_identity.py'
    assert snapshot.read_bytes() == Path(_RUNTIME.__file__).read_bytes()


def test_runtime_change_rejects_qualified_loader_without_modifying_evidence(tmp_path, fake_runtime):
    args, fake = arguments(tmp_path), Scores()
    with synthetic(fake):
        _RUNNER.run_search(args)
    before = output_bytes(args.output_dir)
    fake_runtime['libraries'][0]['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='[Ee]xecution identity|[Rr]untime'):
        _RUNNER.load_qualified_selection(args.output_dir)
    assert output_bytes(args.output_dir) == before


@pytest.mark.parametrize('entry_point', ['resume', 'qualified'])
def test_previous_schema_without_runtime_requires_fresh_directory(tmp_path, entry_point):
    args, fake = arguments(tmp_path), Scores()
    with synthetic(fake):
        _RUNNER.run_search(args)
        path = args.output_dir / 'config.json'
        config = read(path)
        assert config['schema_version'] > 4
        config['schema_version'] = 4
        config['execution_identity'].pop('numerical_runtime')
        path.write_text(json.dumps(config), encoding='utf-8')
        locked_path = args.output_dir / 'qualified-selection.json'
        locked = read(locked_path)
        locked['evidence_sha256']['config.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
        locked_path.write_text(json.dumps(locked), encoding='utf-8')
        before = output_bytes(args.output_dir)
        calls, constructions = copy.deepcopy(fake.calls), len(fake.constructed)
        with pytest.raises(ValueError, match='fresh output directory'):
            if entry_point == 'resume':
                args.resume = True
                _RUNNER.run_search(args)
            else:
                _RUNNER.load_qualified_selection(args.output_dir)
    assert fake.calls == calls and len(fake.constructed) == constructions
    assert output_bytes(args.output_dir) == before


def test_run_and_loader_measure_requested_threads_then_restore_ambient(tmp_path):
    args, fake = arguments(tmp_path, threads=1), Scores()
    original_identity = _RUNNER.execution_identity
    seen = []

    def checked_identity(backend):
        seen.append(pool_threads())
        assert seen[-1] and all(item[-1] == args.threads for item in seen[-1])
        return original_identity(backend)

    with threadpool_limits(2):
        ambient = pool_threads()
        assert any(item[-1] == 2 for item in ambient)
        with patch.object(_RUNNER, 'execution_identity', side_effect=checked_identity), synthetic(fake):
            result = _RUNNER.run_search(args)
            assert pool_threads() == ambient
            saved = read(args.output_dir / 'config.json')['execution_identity']
            assert all(row['execution_identity'] == saved for row in records(args.output_dir))
            args.resume = True
            assert _RUNNER.run_search(args) == result
            assert pool_threads() == ambient
            locked = _RUNNER.load_qualified_selection(args.output_dir)
            assert locked['execution_identity'] == saved
            assert pool_threads() == ambient
    assert len(seen) >= 5


def test_first_thread_limit_in_fresh_process_records_and_restores_all_pools(tmp_path):
    # MKL can lazily load an OpenMP runtime when its first thread limit is set.
    # A fresh interpreter prevents earlier tests from hiding that first use.
    source = Path(__file__).resolve().parents[1]
    script = '''
import json, pathlib, sys
source, destination = map(pathlib.Path, sys.argv[1:3])
sys.path[:0] = [str(source / 'src'), str(source / 'examples'), str(source / 'tests')]
from test_tune_mnist import _RUNNER, Scores, arguments, read, synthetic
from threadpoolctl import threadpool_info, threadpool_limits
args, scores = arguments(destination, threads=1), Scores()
with synthetic(scores):
    result = _RUNNER.run_search(args)
    saved = read(args.output_dir / 'config.json')['execution_identity']['numerical_runtime']
    assert saved['threadpools']
    assert all(pool['num_threads'] == 1 for pool in saved['threadpools']), saved['threadpools']
    with threadpool_limits(4):
        ambient = threadpool_info()
        args.resume = True
        assert _RUNNER.run_search(args) == result
        assert threadpool_info() == ambient
        _RUNNER.load_qualified_selection(args.output_dir)
        assert threadpool_info() == ambient
print(json.dumps({'state': result['state'], 'pools': saved['threadpools']}))
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for name in ('OMP_NUM_THREADS', 'BLIS_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[name] = '4'
    child = subprocess.run([sys.executable, '-B', '-c', script, str(source), str(tmp_path)],
                           env=env, text=True, capture_output=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout)['state'] == 'accepted'


def test_runtime_capture_failure_prevents_new_output_or_fits(tmp_path, monkeypatch):
    args, fake = arguments(tmp_path), Scores()

    def unavailable():
        raise OSError('runtime library cannot be read')

    monkeypatch.setitem(_RUNNER.execution_identity.__globals__, 'runtime_identity', unavailable)
    with synthetic(fake), pytest.raises(OSError, match='cannot be read'):
        _RUNNER.run_search(args)
    assert not args.output_dir.exists() and not fake.calls and not fake.constructed


def test_changed_runtime_prevents_reusing_fold_linear_models(monkeypatch):
    identity = {'runtime': 1}
    monkeypatch.setattr(mnist_cv, 'runtime_identity', lambda: copy.deepcopy(identity))
    evaluator = mnist_cv.FoldEvaluator(*small_data(), max_epochs=2)
    params = {'lambd': .02, 'gamma': .3, 'retention_fraction': .75}
    evaluator.evaluate(params)
    count = evaluator.linear_fit_count_
    identity['runtime'] = 2
    with pytest.raises(RuntimeError, match='execution identity changed'):
        evaluator.evaluate(params)
    assert evaluator.linear_fit_count_ == count
    identity['runtime'] = 1
    restored = evaluator.evaluate(params)
    assert evaluator.linear_fit_count_ == 2 * count
    assert not any(row['linear_cache_hit'] for row in restored['folds'])


@pytest.mark.parametrize('failure', ['changed', 'unreadable'])
def test_runtime_failure_during_fold_fit_discards_uncertain_linear_cache(monkeypatch, failure):
    identity = {'runtime': 1}

    def identify():
        if failure == 'unreadable' and identity['runtime'] == 2:
            raise OSError('runtime file became unreadable during fitting')
        return copy.deepcopy(identity)

    monkeypatch.setattr(mnist_cv, 'runtime_identity', identify)
    evaluator = mnist_cv.FoldEvaluator(*small_data(), max_epochs=2)
    original_fit = mnist_cv.KernelSGDDWD.fit

    def changing_fit(model, *args, **kwargs):
        result = original_fit(model, *args, **kwargs)
        identity['runtime'] = 2
        return result

    monkeypatch.setattr(mnist_cv.KernelSGDDWD, 'fit', changing_fit)
    expected, message = ((RuntimeError, 'execution identity changed') if failure == 'changed'
                         else (OSError, 'unreadable during fitting'))
    with pytest.raises(expected, match=message):
        evaluator.evaluate({'lambd': .02, 'gamma': .3, 'retention_fraction': .75})
    assert evaluator.linear_fit_count_ == 2
    assert not evaluator._linear_cache
    # Restoration cannot make uncertain fits eligible for a later cache hit.
    identity['runtime'] = 1
    monkeypatch.setattr(mnist_cv.KernelSGDDWD, 'fit', original_fit)
    result = evaluator.evaluate({'lambd': .02, 'gamma': .3, 'retention_fraction': .75})
    assert evaluator.linear_fit_count_ == 4
    assert not any(row['linear_cache_hit'] for row in result['folds'])


@pytest.mark.parametrize('failure', [RuntimeError('second kernel failed'), KeyboardInterrupt()])
def test_failed_candidate_before_identity_check_discards_uncertain_linear_cache(monkeypatch, failure):
    identity = {'runtime': 1}
    monkeypatch.setattr(mnist_cv, 'runtime_identity', lambda: copy.deepcopy(identity))
    evaluator = mnist_cv.FoldEvaluator(*small_data(), max_epochs=2)
    original_fit = mnist_cv.KernelSGDDWD.fit
    calls = []

    def interrupted_fit(model, *args, **kwargs):
        calls.append(model)
        if len(calls) == 2:
            raise failure
        result = original_fit(model, *args, **kwargs)
        identity['runtime'] = 2
        return result

    monkeypatch.setattr(mnist_cv.KernelSGDDWD, 'fit', interrupted_fit)
    params = {'lambd': .02, 'gamma': .3, 'retention_fraction': .75}
    with pytest.raises(type(failure)) as caught:
        evaluator.evaluate(params)
    assert caught.value is failure
    assert evaluator.linear_fit_count_ == 2
    assert not evaluator._linear_cache
    identity['runtime'] = 1
    monkeypatch.setattr(mnist_cv.KernelSGDDWD, 'fit', original_fit)
    result = evaluator.evaluate(params)
    assert evaluator.linear_fit_count_ == 4
    assert not any(row['linear_cache_hit'] for row in result['folds'])


def test_record_runtime_mismatch_rejected_before_resume_fit(tmp_path, fake_runtime):
    args, fake = arguments(tmp_path), Scores(fail_once=(1., 1.))
    with synthetic(fake):
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            _RUNNER.run_search(args)
        rows = records(args.output_dir)
        rows[0]['execution_identity']['numerical_runtime']['libraries'][0]['sha256'] = 'b' * 64
        path = args.output_dir / 'records.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        before = output_bytes(args.output_dir)
        calls = copy.deepcopy(fake.calls)
        args.resume = True
        with pytest.raises(ValueError, match='[Ii]dentity'):
            _RUNNER.run_search(args)
    assert fake.calls == calls and output_bytes(args.output_dir) == before
