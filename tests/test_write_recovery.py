"""Atomic JSON failures and secondary diagnostics preserve recoverable evidence."""
import json
import os
from pathlib import Path
from unittest.mock import patch
import warnings

import pytest

from test_tune_mnist import _RUNNER, Scores, arguments, records, synthetic


def _same_path(left, right):
    """Compare actual I/O targets without depending on the production adapter."""
    def key(value):
        value = os.fspath(value)
        if os.name == 'nt':
            if value.startswith('\\\\?\\UNC\\'):
                value = '\\\\' + value[8:]
            elif value.startswith('\\\\?\\'):
                value = value[4:]
        return os.path.normcase(os.path.abspath(value))
    return key(left) == key(right)


class _ObservedStream:
    """Use a real temporary file while injecting one deterministic I/O failure."""

    def __init__(self, stream, events, stage=None, failure=None):
        self.stream = stream
        self.events = events
        self.stage = stage
        self.failure = failure

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        try:
            return self.stream.__exit__(*args)
        finally:
            self.events.append('close')

    def write(self, text):
        self.events.append('write')
        if self.stage == 'write':
            self.stream.write(text[:5])
            raise self.failure
        return self.stream.write(text)

    def flush(self):
        self.events.append('flush')
        if self.stage == 'flush':
            raise self.failure
        return self.stream.flush()

    def fileno(self):
        return self.stream.fileno()


@pytest.mark.parametrize('failure', [
    ValueError('original evaluator error'), KeyboardInterrupt(), SystemExit(7),
])
@pytest.mark.parametrize('blocked', [
    {'failure.json'}, {'status.json'}, {'failure.json', 'status.json'},
])
def test_secondary_diagnostic_failure_preserves_exception_prefix_and_resume(tmp_path, failure, blocked):
    args = arguments(tmp_path)
    fake = Scores(fail_once=(1., 1.), fail_error=failure)
    original_write = _RUNNER.write_json
    diagnostics, committed = [], []

    def deny_diagnostic(path, value):
        if path.name == 'failure.json' or (path.name == 'status.json' and 'error_type' in value):
            diagnostics.append(path.name)
            committed.append((args.output_dir / 'records.jsonl').read_bytes())
            if path.name in blocked:
                raise PermissionError('diagnostic destination unavailable: ' + path.name)
        return original_write(path, value)

    with synthetic(fake):
        # A warning policy must not turn secondary reporting into the new error.
        with warnings.catch_warnings(), patch.object(_RUNNER, 'write_json', side_effect=deny_diagnostic):
            warnings.simplefilter('error')
            with pytest.raises(type(failure)) as caught:
                _RUNNER.run_search(args)
        assert caught.value is failure
        assert diagnostics == ['failure.json', 'status.json']
        assert len(committed) == 2 and committed[0] == committed[1]
        prefix = committed[0]
        assert prefix and len(prefix.splitlines()) == 4
        assert (args.output_dir / 'records.jsonl').read_bytes() == prefix
        saved = records(args.output_dir)
        if 'failure.json' not in blocked:
            diagnostic = json.loads((args.output_dir / 'failure.json').read_text())
            assert diagnostic['error_type'] == type(failure).__name__
            assert diagnostic['completed_candidates'] == len(saved)
        if 'status.json' not in blocked:
            status = json.loads((args.output_dir / 'status.json').read_text())
            assert status['state'] == ('interrupted' if isinstance(failure, (KeyboardInterrupt, SystemExit)) else 'failed')
        # The failed invocation relinquished ownership even if neither report saved.
        with _RUNNER._RunLease(args.output_dir / '.run.lock'):
            pass
        args.resume = True
        result = _RUNNER.run_search(args)
        assert result['state'] == 'accepted'
        assert (args.output_dir / 'records.jsonl').read_bytes().startswith(prefix)
        assert records(args.output_dir)[:len(saved)] == saved
        assert len(fake.search_calls) == result['attempted_candidates'] + 1
        assert sum(params['gamma'] == 'scale' for params in fake.calls) == 1


@pytest.mark.parametrize('secondary', [KeyboardInterrupt(), SystemExit(19)])
def test_new_control_flow_interrupt_during_diagnostics_remains_visible(tmp_path, secondary):
    original = ValueError('original evaluator error')
    args, fake = arguments(tmp_path), Scores(fail_once=(1., 1.), fail_error=original)
    write = _RUNNER.write_json

    def interrupt_diagnostic(path, value):
        if path.name == 'failure.json':
            raise secondary
        return write(path, value)

    with synthetic(fake), patch.object(_RUNNER, 'write_json', side_effect=interrupt_diagnostic):
        with pytest.raises(type(secondary)) as caught:
            _RUNNER.run_search(args)
    assert caught.value is secondary
    with _RUNNER._RunLease(args.output_dir / '.run.lock'):
        pass


@pytest.mark.parametrize('secondary', [KeyboardInterrupt(), SystemExit(23)])
def test_new_interrupt_from_diagnostic_warning_is_not_suppressed(tmp_path, secondary):
    args, fake = arguments(tmp_path), Scores(fail_once=(1., 1.))
    write = _RUNNER.write_json

    def deny_diagnostic(path, value):
        if path.name == 'failure.json':
            raise PermissionError('diagnostic denied')
        return write(path, value)

    with synthetic(fake), patch.object(_RUNNER, 'write_json', side_effect=deny_diagnostic), \
            patch.object(warnings, 'warn', side_effect=secondary):
        with pytest.raises(type(secondary)) as caught:
            _RUNNER.run_search(args)
    assert caught.value is secondary
    with _RUNNER._RunLease(args.output_dir / '.run.lock'):
        pass


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_json_is_rejected_before_touching_destination_or_temporary(tmp_path, value):
    target = tmp_path / 'checkpoint.json'
    temporary = target.with_suffix('.json.tmp')
    target.write_bytes(b'previous checkpoint\n')
    temporary.write_bytes(b'unowned temporary\n')
    opened = []
    original_open = Path.open

    def observe_open(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    with patch.object(Path, 'open', autospec=True, side_effect=observe_open):
        with pytest.raises(ValueError, match='JSON compliant'):
            _RUNNER.write_json(target, {'value': value})
    assert opened == []
    assert target.read_bytes() == b'previous checkpoint\n'
    assert temporary.read_bytes() == b'unowned temporary\n'


@pytest.mark.parametrize('stage,failure', [
    ('write', PermissionError('partial write failed')),
    ('flush', PermissionError('flush failed')),
    ('fsync', PermissionError('fsync failed')),
    ('replace', PermissionError('replace failed')),
    ('fsync', KeyboardInterrupt()),
    ('replace', SystemExit(11)),
])
def test_atomic_failure_removes_owned_temporary_and_preserves_primary(tmp_path, stage, failure):
    target = tmp_path / 'checkpoint.json'
    temporary = target.with_suffix('.json.tmp')
    target.write_bytes(b'previous checkpoint\n')
    events = []
    original_open, original_replace = Path.open, Path.replace
    original_fsync = _RUNNER.os.fsync

    def open_temporary(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return _ObservedStream(stream, events, stage, failure) if _same_path(path, temporary) else stream

    def fsync(descriptor):
        events.append('fsync')
        if stage == 'fsync':
            raise failure
        return original_fsync(descriptor)

    def replace(path, destination):
        events.append('replace')
        if _same_path(path, temporary) and stage == 'replace':
            raise failure
        return original_replace(path, destination)

    with patch.object(Path, 'open', autospec=True, side_effect=open_temporary), \
            patch.object(_RUNNER.os, 'fsync', side_effect=fsync), \
            patch.object(Path, 'replace', autospec=True, side_effect=replace):
        with pytest.raises(type(failure)) as caught:
            _RUNNER.write_json(target, {'new': [1, 2, 3]})
    assert caught.value is failure
    assert 'close' in events
    assert not temporary.exists()
    assert target.read_bytes() == b'previous checkpoint\n'


@pytest.mark.parametrize('preexisting', [False, True])
def test_failed_temporary_acquisition_does_not_unlink_unowned_path(tmp_path, preexisting):
    target = tmp_path / 'checkpoint.json'
    temporary = target.with_suffix('.json.tmp')
    target.write_bytes(b'previous checkpoint\n')
    if preexisting:
        temporary.write_bytes(b'unowned temporary\n')
    failure = PermissionError('temporary acquisition denied')
    original_open, original_unlink = Path.open, Path.unlink

    def deny_open(path, *args, **kwargs):
        if _same_path(path, temporary):
            raise failure
        return original_open(path, *args, **kwargs)

    with patch.object(Path, 'open', autospec=True, side_effect=deny_open), \
            patch.object(Path, 'unlink', autospec=True, side_effect=original_unlink) as unlink:
        with pytest.raises(PermissionError) as caught:
            _RUNNER.write_json(target, {'new': 1})
    assert caught.value is failure
    unlink.assert_not_called()
    assert target.read_bytes() == b'previous checkpoint\n'
    assert temporary.exists() is preexisting
    if preexisting:
        assert temporary.read_bytes() == b'unowned temporary\n'


def test_cleanup_permission_error_does_not_replace_primary_error(tmp_path):
    target = tmp_path / 'checkpoint.json'
    temporary = target.with_suffix('.json.tmp')
    target.write_bytes(b'previous checkpoint\n')
    primary, cleanup = PermissionError('replace denied'), PermissionError('cleanup denied')
    original_unlink = Path.unlink

    def deny_cleanup(path, *args, **kwargs):
        if _same_path(path, temporary):
            raise cleanup
        return original_unlink(path, *args, **kwargs)

    with patch.object(Path, 'replace', autospec=True, side_effect=primary), \
            patch.object(Path, 'unlink', autospec=True, side_effect=deny_cleanup) as unlink:
        with pytest.raises(PermissionError) as caught:
            _RUNNER.write_json(target, {'new': 1})
    assert caught.value is primary
    assert unlink.call_count == 1
    assert temporary.exists()
    assert target.read_bytes() == b'previous checkpoint\n'


@pytest.mark.parametrize('secondary', [KeyboardInterrupt(), SystemExit(29)])
def test_new_interrupt_during_temporary_cleanup_is_not_suppressed(tmp_path, secondary):
    target = tmp_path / 'checkpoint.json'
    temporary = target.with_suffix('.json.tmp')
    target.write_bytes(b'previous checkpoint\n')
    primary = PermissionError('replace denied')
    with patch.object(Path, 'replace', autospec=True, side_effect=primary), \
            patch.object(Path, 'unlink', autospec=True, side_effect=secondary):
        with pytest.raises(type(secondary)) as caught:
            _RUNNER.write_json(target, {'new': 1})
    assert caught.value is secondary
    assert temporary.exists()
    assert target.read_bytes() == b'previous checkpoint\n'


def test_successful_write_flushes_and_syncs_before_closed_file_replace(tmp_path):
    target = tmp_path / 'checkpoint.json'
    temporary = target.with_suffix('.json.tmp')
    target.write_bytes(b'previous checkpoint\n')
    value, events = {'values': [1, True, None], 'name': 'lambda'}, []
    original_open, original_replace = Path.open, Path.replace
    original_fsync = _RUNNER.os.fsync
    opened = []

    def open_temporary(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if _same_path(path, temporary):
            opened.append(stream)
            events.append('open')
            return _ObservedStream(stream, events)
        return stream

    def fsync(descriptor):
        assert descriptor == opened[0].fileno()
        events.append('fsync')
        return original_fsync(descriptor)

    def replace(path, destination):
        assert _same_path(path, temporary) and _same_path(destination, target)
        assert opened[0].closed
        events.append('replace')
        return original_replace(path, destination)

    with patch.object(Path, 'open', autospec=True, side_effect=open_temporary), \
            patch.object(_RUNNER.os, 'fsync', side_effect=fsync), \
            patch.object(Path, 'replace', autospec=True, side_effect=replace):
        _RUNNER.write_json(target, value)
    assert events == ['open', 'write', 'flush', 'fsync', 'close', 'replace']
    expected = (json.dumps(value, indent=2, allow_nan=False) + '\n').replace('\n', os.linesep)
    assert target.read_bytes() == expected.encode('utf-8')
    assert json.loads(target.read_text()) == value
    assert not temporary.exists()
