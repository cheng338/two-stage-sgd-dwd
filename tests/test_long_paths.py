"""Real Windows output paths beyond MAX_PATH retain strict search recovery."""
import copy
import hashlib
import json
import ntpath
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from test_tune_mnist import _RUNNER, Scores, arguments, synthetic


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Real Windows long-path qualification')


def _safe(path):
    """Independent setup/assertion adapter; never call the code under test."""
    value = os.fspath(path)
    if value.startswith(('\\\\?\\', '\\\\.\\')):
        return Path(value)
    absolute = ntpath.abspath(value)
    return Path('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\')
                else '\\\\?\\' + absolute)


def _target(tmp_path, length, name):
    parent = tmp_path / 'space é'
    remaining = length - len(str(parent)) - 1 - len(name)
    assert remaining >= 0, 'Use a shorter pytest basetemp to retain the declared path length'
    while remaining > 80:
        component = 'd' * min(64, remaining - 1)
        parent /= component
        remaining -= len(component) + 1
    target = parent / ('p' * remaining + name)
    assert len(str(target)) == length
    assert not str(target).startswith('\\\\?\\')
    _safe(parent).mkdir(parents=True, exist_ok=True)
    return target


def _records(directory):
    return [json.loads(line) for line in _safe(directory / 'records.jsonl').read_bytes().splitlines()]


def _bytes(directory):
    root = _safe(directory)
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file() and path.name != '.run.lock'}


@pytest.mark.parametrize('length', [257, 291])
def test_atomic_json_reads_hashes_and_temporary_suffix_cross_long_path_boundary(tmp_path, length):
    target = _target(tmp_path, length, 'checkpoint.json')
    temporary = target.with_suffix('.json.tmp')
    assert len(str(temporary)) == length + 4
    value = {'finite': [1, 2.5], 'label': 'space é'}
    original = str(target)
    _RUNNER.write_json(target, value)
    expected = (json.dumps(value, indent=2, allow_nan=False) + '\n').replace('\n', os.linesep).encode('utf-8')
    assert _safe(target).read_bytes() == expected
    assert _RUNNER._read_json(target) == value
    assert _RUNNER._file_hash(target) == hashlib.sha256(expected).hexdigest()
    assert not _safe(temporary).exists()
    assert str(target) == original


def test_long_atomic_replace_failure_keeps_primary_and_cleans_owned_temporary(tmp_path):
    target = _target(tmp_path, 291, 'checkpoint.json')
    temporary = target.with_suffix('.json.tmp')
    _safe(target).write_bytes(b'previous checkpoint\n')
    primary = PermissionError('replacement denied')
    replace = Path.replace
    attempted = []

    def deny(path, destination):
        if _safe(path) == _safe(temporary):
            attempted.append((path, destination))
            raise primary
        return replace(path, destination)

    with patch.object(Path, 'replace', autospec=True, side_effect=deny):
        with pytest.raises(PermissionError) as caught:
            _RUNNER.write_json(target, {'new': 1})
    assert caught.value is primary and len(attempted) == 1
    assert _safe(target).read_bytes() == b'previous checkpoint\n'
    assert not _safe(temporary).exists()


@pytest.mark.parametrize('spelling', ['absolute', 'relative_parent', 'already_extended'])
def test_fresh_deep_run_completed_replay_and_qualified_load_preserve_caller_spelling(tmp_path, monkeypatch, spelling):
    directory = _target(tmp_path, 291, 'run')
    if spelling == 'relative_parent':
        anchor = tmp_path / 'anchor'
        anchor.mkdir()
        monkeypatch.chdir(tmp_path)
        supplied = Path('anchor') / '..' / directory.relative_to(tmp_path)
    elif spelling == 'already_extended':
        supplied = _safe(directory)
    else:
        supplied = directory
    args, fake = arguments(tmp_path, output_dir=supplied), Scores()
    original = vars(args).copy()
    with synthetic(fake) as (loader, _):
        result = _RUNNER.run_search(args)
        assert result['state'] == 'accepted'
        assert vars(args) == original
        before = _bytes(directory)
        calls, constructors = copy.deepcopy(fake.calls), len(fake.constructed)
        args.resume = True
        assert _RUNNER.run_search(args) == result
        locked = _RUNNER.load_qualified_selection(supplied)
        assert locked['params'] == result['params']
        assert locked['epochs'] == args.epochs and result['mean_cv_accuracy'] == .9
        assert fake.calls == calls and len(fake.constructed) == constructors
        assert loader.call_args.kwargs == {'partition': 'train'}
    assert args.output_dir is supplied and str(args.output_dir) == str(original['output_dir'])
    assert _bytes(directory) == before
    config = json.loads(_safe(directory / 'config.json').read_text(encoding='utf-8'))
    assert all(Path(name).name == name and '\\' not in name and '/' not in name
               for name in config['implementation_sha256'])
    assert len(str(directory / 'implementation-snapshot/tune_mnist.py')) > 260


def test_deep_evaluator_failure_survives_unwritable_diagnostic_then_resumes(tmp_path):
    directory = _target(tmp_path, 291, 'run')
    original = ValueError('original candidate failure')
    args, fake = arguments(tmp_path, output_dir=directory), Scores(fail_once=(1., 1.), fail_error=original)
    write = _RUNNER.write_json
    denied = []

    def deny_failure(path, value):
        if path.name == 'failure.json':
            denied.append(path)
            raise PermissionError('diagnostic denied')
        return write(path, value)

    with synthetic(fake):
        with patch.object(_RUNNER, 'write_json', side_effect=deny_failure), \
                pytest.warns(RuntimeWarning, match='failure.json'):
            with pytest.raises(ValueError) as caught:
                _RUNNER.run_search(args)
        assert caught.value is original and len(denied) == 1
        prefix = _safe(directory / 'records.jsonl').read_bytes()
        assert len(prefix.splitlines()) == 4
        status = json.loads(_safe(directory / 'status.json').read_text(encoding='utf-8'))
        assert status['state'] == 'failed' and status['error_type'] == 'ValueError'
        with _RUNNER._RunLease(directory / '.run.lock'):
            pass
        args.resume = True
        result = _RUNNER.run_search(args)
    assert result['state'] == 'accepted'
    assert _safe(directory / 'records.jsonl').read_bytes().startswith(prefix)
    assert len(fake.search_calls) == result['attempted_candidates'] + 1
    assert args.output_dir is directory


@pytest.mark.parametrize('tail', ['partial', 'complete_without_newline', 'bad_receipt_without_newline',
                                 'malformed_newline', 'bad_prefix_and_partial'])
def test_deep_ledger_tail_recovery_and_refusal_keep_original_evidence(tmp_path, tail):
    directory = _target(tmp_path, 291, 'run')
    args, fake = arguments(tmp_path, output_dir=directory), Scores(fail_once=(1., 1.))
    with synthetic(fake):
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            _RUNNER.run_search(args)
        ledger = _safe(directory / 'records.jsonl')
        prefix, saved = ledger.read_bytes(), _records(directory)
        assert len(saved) == 4
        if tail == 'partial':
            changed = prefix + b'{"params":'
        elif tail == 'complete_without_newline':
            changed = prefix.rstrip(b'\r\n')
        elif tail == 'malformed_newline':
            changed = prefix + b'{"params":\n'
        else:
            edited = copy.deepcopy(saved)
            edited[0 if tail == 'bad_prefix_and_partial' else -1]['_receipt']['sha256'] = '0' * 64
            changed = b'\n'.join(json.dumps(row).encode('utf-8') for row in edited)
            if tail == 'bad_prefix_and_partial':
                changed += b'\n{"params":'
        ledger.write_bytes(changed)
        before, calls = _bytes(directory), copy.deepcopy(fake.calls)
        args.resume = True
        if tail in {'partial', 'complete_without_newline'}:
            result = _RUNNER.run_search(args)
            assert result['state'] == 'accepted'
            assert _records(directory)[:len(saved)] == saved
            assert len(fake.search_calls) == result['attempted_candidates'] + 1
            backup = _safe(directory / 'records-interrupted.jsonl')
            if tail == 'partial':
                assert backup.read_bytes() == changed
            else:
                assert not backup.exists()
                assert ledger.read_bytes().startswith(changed)
        else:
            with pytest.raises(ValueError):
                _RUNNER.run_search(args)
            assert fake.calls == calls and _bytes(directory) == before
            assert not _safe(directory / 'records-interrupted.jsonl').exists()
    assert args.output_dir is directory


def test_snapshot_descendant_past_boundary_rejects_tampering_without_refit(tmp_path):
    # The output directory itself is below MAX_PATH; its snapshot child is not.
    directory = _target(tmp_path, 242, 'run')
    snapshot = directory / 'implementation-snapshot/tune_mnist.py'
    assert len(str(directory)) < 260 < len(str(snapshot))
    args, fake = arguments(tmp_path, output_dir=directory), Scores()
    with synthetic(fake):
        assert _RUNNER.run_search(args)['state'] == 'accepted'
        _safe(snapshot).write_bytes(_safe(snapshot).read_bytes() + b'\n# changed snapshot\n')
        before, calls = _bytes(directory), copy.deepcopy(fake.calls)
        with pytest.raises(ValueError, match='snapshot'):
            _RUNNER.load_qualified_selection(directory)
        args.resume = True
        with pytest.raises(ValueError, match='snapshot'):
            _RUNNER.run_search(args)
        assert fake.calls == calls and _bytes(directory) == before


def test_long_lease_excludes_second_process_then_releases(tmp_path):
    lease_path = _target(tmp_path, 291, '.run.lock')
    script = (
        'import json,sys\n'
        'sys.path[:0]=json.loads(sys.argv[1])\n'
        'from pathlib import Path\n'
        'from test_tune_mnist import _RUNNER\n'
        'try:\n'
        '    with _RUNNER._RunLease(Path(sys.argv[2])):\n'
        '        state="acquired"\n'
        'except _RUNNER.SearchBusyError:\n'
        '    state="busy"\n'
        'print(json.dumps({"state":state}))\n'
    )

    def contender():
        result = subprocess.run([sys.executable, '-B', '-c', script,
                                 json.dumps([str(value) for value in sys.path]), str(lease_path)],
                                capture_output=True, text=True, encoding='utf-8', timeout=60,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)['state']

    lease = _RUNNER._RunLease(lease_path)
    assert lease.path == lease_path
    with lease:
        assert contender() == 'busy'
        with pytest.raises(_RUNNER.SearchBusyError):
            with _RUNNER._RunLease(lease_path):
                pytest.fail('A second handle acquired the live lease')
    assert contender() == 'acquired'
    assert lease.path == lease_path and _safe(lease_path).exists()


def test_io_path_spelling_controls_and_unc_conversion_are_lexical(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ordinary = [tmp_path / 'space é' / 'file.json', Path('space é') / '..' / 'file.json',
                Path('\\root relative\\file.json')]
    for value in ordinary:
        before = str(value)
        assert _RUNNER._io_path(value) == _safe(value)
        assert str(value) == before
    extended = _safe(tmp_path / 'space é' / 'file.json')
    explicit_parent = Path(str(extended.parent) + '\\..\\file.json')
    device = Path('\\\\.\\C:\\example')
    for value in (extended, explicit_parent, device):
        assert _RUNNER._io_path(value) == value
        assert _RUNNER._io_path(_RUNNER._io_path(value)) == value
    # This checks namespace conversion only; it never contacts a UNC share.
    unc = Path('\\\\server\\share\\space é\\file.json')
    assert str(_RUNNER._io_path(unc)) == '\\\\?\\UNC\\server\\share\\space é\\file.json'


def test_non_windows_adapter_returns_the_original_path_form(tmp_path):
    value = Path('space é') / '..' / 'file.json'
    # Replace the subject's OS reference, not global os.name used by pathlib.
    with patch.object(_RUNNER, 'os', SimpleNamespace(name='posix')):
        assert _RUNNER._io_path(value) == value
